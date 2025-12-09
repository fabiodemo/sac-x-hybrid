import rclpy
import argparse
import yaml
import time
import numpy as np
import pandas as pd
import torch
import random

from tools import make_agent, make_env
from model_free.util.buffer import ReplayBuffer

import subprocess
import random
import os
import signal
import time

def compute_competence(episode_rewards, window=50):
    """
    Retorna a média dos últimos 'window' episódios como métrica de competência.
    """
    if len(episode_rewards) < window:
        return np.mean(episode_rewards)
    else:
        return np.mean(episode_rewards[-window:])

def compute_competence_progress(episode_rewards, window=50):
    """
    Computa a diferença entre a média móvel mais recente e a de 1 janela anterior.
    """
    current_competence = compute_competence(episode_rewards, window)
    if len(episode_rewards) >= 2*window:
        past_competence = np.mean(episode_rewards[-2*window:-window])
    else:
        past_competence = 0
    return current_competence - past_competence

def adaptive_alpha_update(agent, competence_progress, alpha_min=0.0001, alpha_max=0.01, factor=0.0005):
    """
    Ajusta alpha do SAC de forma adaptativa com base no progresso de competência.
    - competence_progress > 0 : reduz alpha (menos exploração)
    - competence_progress < 0 : aumenta alpha (mais exploração)
    """
    old_alpha = agent.alpha
    new_alpha = old_alpha - factor * competence_progress
    new_alpha = np.clip(new_alpha, alpha_min, alpha_max)
    agent.alpha = new_alpha
    print(f"[Adaptive α] Old alpha={old_alpha:.6f}, new alpha={agent.alpha:.6f}")



GAZEBO_WORLDS = [
        "/home/fabio/ros2_humble/src/turtlebot3_simulations/turtlebot3_gazebo/worlds/turtlebot3_dqn_stage1/burger.model",
        "/home/fabio/ros2_humble/src/turtlebot3_simulations/turtlebot3_gazebo/worlds/turtlebot3_dqn_stage2/burger.model",
        "/home/fabio/ros2_humble/src/turtlebot3_simulations/turtlebot3_gazebo/worlds/turtlebot3_dqn_stage4/burger.model",
        "/home/fabio/ros2_humble/src/turtlebot3_simulations/turtlebot3_gazebo/worlds/turtlebot3_dqn_stage9/burger.model",
        # "/home/fabio/ros2_humble/src/turtlebot3_simulations/turtlebot3_gazebo/worlds/turtlebot3_dqn_open_world_1/burger.model",
        # "/home/fabio/ros2_humble/src/turtlebot3_simulations/turtlebot3_gazebo/worlds/turtlebot3_dqn_open_world_3/burger.model",
]

STEP_SYNC = 0.15

configs = {}

def transfer_experience(source_buffer, target_buffer, fraction=0.1):
    """
    Transfere uma fração das transições do buffer de origem para o buffer de destino.
    """
    num_samples = int(source_buffer.mem_cntr * fraction)
    if num_samples > 0:
        indices = np.random.choice(source_buffer.mem_cntr, num_samples, replace=False)
        for idx in indices:
            state = source_buffer.state_memory[idx]
            action = source_buffer.action_memory[idx]
            reward = source_buffer.reward_memory[idx]
            new_state = source_buffer.new_state_memory[idx]
            done = source_buffer.terminal_memory[idx]
            target_buffer.store_transition(state, action, reward, new_state, done)

def get_stage_from_world_path(world_path):
    """Retorna o número do stage baseado no nome do mundo."""
    # Dicionário de mapeamento de trechos do caminho para o stage correspondente
    stage_mapping = {
        "stage1": 1,
        "stage2": 2,
        "stage4": 3,
        "stage9": 4,
        "open_world_1": 5,
        "open_world_3": 6,
    }
    for key, stage in stage_mapping.items():
        if key in world_path:
            return stage
    
    raise ValueError(f"Não foi possível mapear o stage para o caminho: {world_path}")

def get_world_queue():
    """Gera uma fila ordenada de mundos."""
    return GAZEBO_WORLDS[:]

def launch_gazebo(world_path):
    """Launch Gazebo with a specified world."""
    print(f"Launching Gazebo world: {world_path}")
    try:
        print(world_path)
        command = [
            'ros2', 'launch', 'gazebo_ros', 'gazebo.launch.py',
            f'world:={world_path}'
        ]
        process = subprocess.Popen(command)
        time.sleep(10)
        return process
    except Exception as e:
        print(f"Error launching Gazebo: {e}")
        return None

def kill_gazebo(process):
    """Terminate the running Gazebo simulation."""
    if process:
        try:
            os.kill(process.pid, signal.SIGINT)
            print("Gazebo simulation stopped.")
            time.sleep(10)
        except Exception as e:
            print(f"Error stopping Gazebo: {e}")

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_csv_progress(file_path="train.csv"):
    """Carrega progresso de treinamento a partir de um arquivo CSV, se existir."""
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        last_episode = df.index[-1]  # Último índice (episódio)
        acum_rwds = df['scores'].tolist()  # Carrega a lista de recompensas acumuladas
        steps_rwds = df['steps'].tolist()  # Carrega a lista de passos
        print(f"Resuming from episode {last_episode + 1}")
        return last_episode + 1, acum_rwds, steps_rwds
    else:
        print("No previous progress found. Starting from episode 0.")
        return 0, [], []

def train():
    # Inicializa valores de progresso a partir do CSV
    start_episode, acum_rwds, steps_rwds = load_csv_progress('train.csv')

    best_moving_average = -np.inf
    n_steps = 0 if not steps_rwds else steps_rwds[-1]
    comp_progress = 0.0
    alpha_history = []
    comp_progress_history = []
    actor_loss_history = []
    critic_loss_history = []
    current_stage_history = []

    world_queue = get_world_queue()
    save_csv_every = 50
    total_episodes = 5000
    episodes_per_world = total_episodes // len(world_queue)

    print(f'Training for {total_episodes} episodes, {episodes_per_world} per world')

    success_threshold = 0.85  # Taxa de sucesso mínima para progredir
    reward_threshold = 85    # Recompensa acumulada média mínima
    evaluation_interval = 10  # Intervalo de episódios para avaliar competência
    
    for world_path in world_queue:
        stage = get_stage_from_world_path(world_path)
        print(f"Training on world: {world_path}, Stage: {stage}")
        gazebo_process = launch_gazebo(world_path)
        env = make_env(stage, configs['max_steps_per_episode'], configs['lidar'])

        # Calcular input_dims dinamicamente e atualizar configs
        configs['input_dims'] = (
            env.observation_space.spaces['sensor_readings'].shape[0] +
            env.observation_space.spaces['target'].shape[0] +
            env.observation_space.spaces['velocity'].shape[0]
        )

        # Inicialize o ReplayBuffer para o estágio atual
        stage_replay_buffer = ReplayBuffer(
            max_size=100000,
            input_shape=configs['input_dims'],
            n_actions=env.action_space.shape[0]
        )

        agent = make_agent(env, configs, shared_memory=stage_replay_buffer)
        agent.load_models()

        #####################################################
        for _ in range(100):  # Episódios de exploração
            obs_dict = env.reset()
            obs = obs_dict['sensor_readings'] + obs_dict['target'] + obs_dict['velocity']
            done = False
            while not done:
                action = np.random.uniform(-1, 1, env.action_space.shape[0])  # Ações aleatórias
                obs_dict_, reward, done, _ = env.step({'action': action})
                obs_ = obs_dict_['sensor_readings'] + obs_dict_['target'] + obs_dict_['velocity']
                stage_replay_buffer.store_transition(obs, action, reward, obs_, done)
                obs = obs_

        #####################################################


        # Agora podemos acessar o checkpoint_dir do agent
        fpath = os.path.join(agent.checkpoint_dir, 'train.csv')

        # Listas para armazenar métricas do estágio atual
        stage_rewards = []
        stage_successes = []

        for episode in range(start_episode, episodes_per_world):
            step = 0
            done = False
            acum_reward = 0

            obs_dict = env.reset()
            obs = obs_dict['sensor_readings'] + obs_dict['target'] + obs_dict['velocity']

            while not done:
                step_start = time.time()
                action_dict = {'action': agent.choose_action(obs)}
                obs_dict_, reward, done, _ = env.step(action_dict)
                obs_ = obs_dict_['sensor_readings'] + obs_dict_['target'] + obs_dict_['velocity']
                agent.remember(obs, action_dict['action'], reward, obs_, done)
                obs = obs_
                acum_reward += reward
                losses = agent.learn()

                step += 1
                n_steps += 1

                elapsed = time.time() - step_start
                time.sleep(STEP_SYNC - elapsed if elapsed < STEP_SYNC else 0)

            print(f"Episode {episode} * Accumulated Reward is ==> {acum_reward}")
            if losses is not None:
                critic_loss, actor_loss = losses
            else:
                critic_loss, actor_loss = 0.0, 0.0

            # Salvar métricas de perda
            if isinstance(critic_loss, torch.Tensor):
                critic_loss = critic_loss.item()
            if isinstance(actor_loss, torch.Tensor):
                actor_loss = actor_loss.item()

            critic_loss_history.append(critic_loss)
            actor_loss_history.append(actor_loss)
            acum_rwds.append(acum_reward)
            steps_rwds.append(n_steps)
            comp_progress_history.append(comp_progress)
            alpha_history.append(agent.alpha)
            current_stage_history.append(stage)

            # Armazena as recompensas e sucessos do estágio atual
            stage_rewards.append(acum_reward)
            if acum_reward >= reward_threshold:
                stage_successes.append(1)
            else:
                stage_successes.append(0)

            # Avaliação periódica
            if (episode + 1) % evaluation_interval == 0:
                mean_reward = np.mean(stage_rewards[-evaluation_interval:])
                success_rate = np.mean(stage_successes[-evaluation_interval:])
                print(f"Stage {stage} - Episode {episode + 1}: Mean Reward = {mean_reward:.2f}, Success Rate = {success_rate:.2f}")

                agent.log_stage_metrics(stage, episode + 1, mean_reward, success_rate)

                # Progresso para o próximo estágio
                if mean_reward > reward_threshold and success_rate > success_threshold:
                    # transfer_experience(agent.memory, stage_replay_buffer, fraction=0.3)
                    print(f"Progressing to the next stage after {episode + 1} episodes.")
                    break  # Sai do loop de episódios e avança para o próximo estágio

            # Adaptação de alpha baseada no progresso
            if episode % 50 == 0 and episode > 0:
                comp_progress = compute_competence_progress(acum_rwds, window=50)
                adaptive_alpha_update(agent, comp_progress)

            # Atualizar checkpoint do modelo
            if episode >= save_csv_every - 1:
                moving_avg = np.mean(acum_rwds[-save_csv_every:])
                if moving_avg > best_moving_average:
                    best_moving_average = moving_avg
                    agent.save_models()
                    print(f"Saving best models with moving average reward {best_moving_average}...")

   

        # Fechar ambiente e Gazebo ao final do estágio
        env.close()
        kill_gazebo(gazebo_process)
        time.sleep(10)


    df = pd.DataFrame({
        'episode': range(len(acum_rwds)), 
        'scores': acum_rwds,
        'steps': steps_rwds,
        'alpha': alpha_history,
        'comp_progress': comp_progress_history,
        'critic_loss': critic_loss_history,
        'actor_loss': actor_loss_history,
        'stage': current_stage_history
    })
    df.index.name = 'episode'
    df.to_csv(f'{fpath}', mode='w')

    return acum_rwds, steps_rwds





def main(args=None):
    rclpy.init()
    # env = make_env(configs['stage'], configs['max_steps_per_episode'], configs['lidar'])
    # agent = make_agent(env, configs)
    
    _, _ = train(
        # agent=agent,
        # env=env
        )
    
    env.close()
    rclpy.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Agent on TurtleBot3 Navigation Environment')
    parser.add_argument('--agent', type=str, default='sac', help='Specify the RL agent (sac, ddpg, td3, sac_x_hybrid, sac_x)')
    parser.add_argument('--stage', type=str, default=1, help='Specify the environment stage: 1, 2, 3, 4')
    parser.add_argument('--load', type=bool, default=False, help='Load agent network models: True, False')
    parser.add_argument('--lidar', type=int, default=0, help='Specify the number of LIDAR readings: 10, 360')
    args = parser.parse_args()

    configs = {}

    configs['agent'] = args.agent
    configs['stage'] = args.stage
    configs['load_models'] = args.load
    configs['test'] = False
    configs['lidar'] = args.lidar

    with open('configs.yaml', 'r') as file:
        config_data = yaml.safe_load(file)

        if args.agent in config_data:
            configs.update(config_data[args.agent])
            configs['train_episodes'] = config_data[args.agent].get('train_episodes', 5001)
            configs['max_steps_per_episode'] = config_data[args.agent].get('max_steps_per_episode', 500) # 500 for stage 6
        else:
            raise ValueError(f"No configuration found for agent: {args.agent}")

    set_all_seeds()
    main()