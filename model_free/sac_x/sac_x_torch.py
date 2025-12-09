import os
import torch as T
import torch.nn.functional as F
import numpy as np
from model_free.util.buffer import ReplayBuffer
from model_free.sac.networks import ActorNetwork, CriticNetwork, ValueNetwork
from .skills_policy import SkillPolicy
from .rewards import reward_avoid_obstacles, reward_follow_walls, reward_efficient_exploration
from .meta_controller import MetaController

class Agent:
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=0, max_action=0, 
                 gamma=0.99, n_actions=2, max_size=100000, tau=0.001, batch_size=128,
                 reward_scale=2, min_action=0, checkpoint_dir='tmp/sac_x', num_skills=3):
        """
        Initialize the Soft Actor-Critic agent.

        Args:
            alpha (float): Learning rate for the actor network.
            beta (float): Learning rate for the critic and value networks.
            input_dims (int): Dimensions of the input state.
            max_action (float): Maximum magnitude of action.
            gamma (float): Discount factor for future rewards.
            n_actions (int): Number of actions.
            max_size (int): Maximum size of the replay buffer.
            tau (float): Soft update coefficient for target networks.
            batch_size (int): Size of the batch for learning.
            reward_scale (float): Scaling factor for rewards.
            min_action (float): Minimum magnitude of action.
            checkpoint_dir (str): Directory to save the model checkpoints.
        """
        self.checkpoint_dir = checkpoint_dir
        self.gamma = float(gamma.strip(','))  # Garantir que gamma seja um float
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.name = 'sac'
        self.scale = reward_scale
        self.visited_states = {} 
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.actor = ActorNetwork(alpha, input_dims, max_action=max_action,
                                  n_actions=n_actions, name='actor', chkpt_dir=checkpoint_dir)
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                                      name='critic_1', chkpt_dir=checkpoint_dir)
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                                      name='critic_2', chkpt_dir=checkpoint_dir)
        self.value = ValueNetwork(beta, input_dims, name='value', chkpt_dir=checkpoint_dir)
        self.target_value = ValueNetwork(beta, input_dims, name='target_value', chkpt_dir=checkpoint_dir)

        self.update_network_parameters(tau=1)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.skills = [SkillPolicy(input_dims=input_dims, n_actions=n_actions) for _ in range(num_skills)]
        self.meta_controller = MetaController(input_dims=input_dims, num_skills=num_skills)

    def choose_action(self, observation):
        """
        Choose an action based on the current observation.

        Args:
            observation: The current state observation.

        Returns:
            action: The action chosen by the agent.
        """
        state = T.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        """
        Store a transition in the agent's memory.

        Args:
            state: The starting state.
            action: The action taken.
            reward: The reward received.
            new_state: The resulting state after the action.
            done: Boolean indicating whether the episode is finished.
        """
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        """
        Perform a soft update of the target network parameters.

        Args:
            tau (float, optional): The update factor. If None, use self.tau.
        """
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_dict = dict(target_value_params)
        value_dict = dict(value_params)

        for name in value_dict:
            value_dict[name] = tau * value_dict[name].clone() + (1 - tau) * target_value_dict[name].clone()

        self.target_value.load_state_dict(value_dict)

    def save_models(self):
        """
        Save the current state of all networks as checkpoints.
        """
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        """
        Load the saved states of all networks from checkpoints.
        """
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        """
        Train the agent using experiences from the replay buffer. Decides which auxiliary policy to use
        via the meta-controller and updates networks accordingly.
        """
        if self.memory.mem_cntr < self.batch_size:
            return None  # Verifica se há amostras suficientes para o treinamento

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        state = T.tensor(state, dtype=T.float32).to(self.device)
        new_state = T.tensor(new_state, dtype=T.float32).to(self.device)
        action = T.tensor(action, dtype=T.float32).to(self.device)
        reward = T.tensor(reward, dtype=T.float32).to(self.device)
        done = T.tensor(done, dtype=T.bool).to(self.device)
        meta_loss = 0

        current_policy_indices = self.meta_controller.decide_policy(state)
        
        if len(current_policy_indices.shape) == 0:  # Manipula o tensor se for um único elemento
            current_policy_indices = current_policy_indices.unsqueeze(0)
        
        losses = []
        for idx, current_policy_index in enumerate(current_policy_indices):
            current_policy = self.skills[current_policy_index.item()]  # Obtém a política de habilidade correspondente
            
            # Extrai fatias específicas para processamento de etapa única
            s, a, r, ns, d = state[idx], action[idx], reward[idx], new_state[idx], done[idx]
            s = s.unsqueeze(0) if s.dim() == 1 else s
            a = a.unsqueeze(0) if a.dim() == 1 else a
            ns = ns.unsqueeze(0) if ns.dim() == 1 else ns
            d = d.unsqueeze(0) if d.dim() == 0 else d
            r = r.unsqueeze(0) if r.dim() == 0 else r

            # Calcula recompensas intrínsecas para a política escolhida
            intrinsic_reward = current_policy.compute_intrinsic_reward(s, a, ns, d)

            # Combina recompensas intrínsecas e extrínsecas para sinal de treinamento total
            total_reward = r + intrinsic_reward
            total_reward = total_reward.unsqueeze(0) if total_reward.dim() == 0 else total_reward

            # Atualiza as redes críticas
            self.critic_1.optimizer.zero_grad()
            self.critic_2.optimizer.zero_grad()

            # Convertendo d para float e garantindo a forma correta
            d_float = d.float().view(-1, 1)

            target_value_ns = self.target_value(ns).view(-1)
            one_minus_d_float = (1 - d_float.view(-1))
            total_reward_view = total_reward.view(-1)

            # Corrigindo o cálculo do q_hat para evitar o erro de indexação
            # Parte 1: gamma * target_value_ns
            try:
                part1 = self.gamma * target_value_ns
            except Exception as e:
                print("Error in part1 calculation:", str(e))

            # Parte 2: (gamma * target_value_ns) * one_minus_d_float
            try:
                part2 = part1 * one_minus_d_float
            except Exception as e:
                print("Error in part2 calculation:", str(e))

            # Parte 3: (gamma * target_value_ns * one_minus_d_float) + total_reward_view
            try:
                q_hat = part2 + total_reward_view
            except Exception as e:
                print("Error in q_hat calculation:", str(e))

            q1_old_policy = self.critic_1.forward(s, a).view(-1)
            q2_old_policy = self.critic_2.forward(s, a).view(-1)

            critic_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat) + 0.5 * F.mse_loss(q2_old_policy, q_hat)
            critic_loss.backward()
            self.critic_1.optimizer.step()
            self.critic_2.optimizer.step()

            # Atualiza a rede do ator
            self.actor.optimizer.zero_grad()
            pred_actions, log_probs = self.actor.sample_normal(s, reparameterize=True)
            log_probs = log_probs.view(-1)
            q1_new_policy = self.critic_1.forward(s, pred_actions)
            q2_new_policy = self.critic_2.forward(s, pred_actions)
            critic_value = T.min(q1_new_policy, q2_new_policy).view(-1)

            actor_loss = (log_probs - critic_value).mean()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Atualização suave das redes-alvo
            self.update_network_parameters()

            # Treina o meta-controlador
            meta_loss = self.meta_controller.learn(s, total_reward, current_policy_index)
            if meta_loss is not None:  # Verifica se meta_loss não é None
                losses.append((critic_loss.item(), actor_loss.item(), meta_loss.item()))  # Usa .item() se não for None
            else:
                losses.append((critic_loss.item(), actor_loss.item(), 0))  # Usa um valor padrão se meta_loss for None

        # Agrega perdas para relatórios ou depuração
        return meta_loss

    
    def update_skill_policies(self, states, skill_rewards):
        for idx, skill in enumerate(self.skills):
            skill_optimizer = skill.optimizer
            skill_loss = self.compute_skill_loss(skill, states, skill_rewards[idx])
            skill_optimizer.zero_grad()
            skill_loss.backward()
            skill_optimizer.step()

    def compute_skill_loss(self, skill, states, intrinsic_rewards):
        # Convert intrinsic_rewards to a tensor if it's a list
        if isinstance(intrinsic_rewards, list):
            intrinsic_rewards = T.tensor(intrinsic_rewards, dtype=T.float).to(self.actor.device)

        predicted_actions = skill(states)

        # Aggregate intrinsic rewards by taking the mean or sum
        aggregated_rewards = T.mean(intrinsic_rewards, dim=0, keepdim=True)  # Reduces to size [1]

        # Expand the aggregated rewards to match the batch size and action dimension
        expanded_rewards = aggregated_rewards.expand(predicted_actions.size(0), predicted_actions.size(1))

        # Compute skill loss
        skill_loss = -T.mean(predicted_actions * expanded_rewards)
        return skill_loss


