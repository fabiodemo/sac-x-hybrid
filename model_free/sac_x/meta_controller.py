import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MetaController(nn.Module):
    def __init__(self, input_dims, num_skills, fc1_dims=256, fc2_dims=256, alpha=0.0003, chkpt_dir='tmp/sac_x'):
        super(MetaController, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.policy = nn.Linear(fc2_dims, num_skills)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.checkpoint_file = os.path.join(chkpt_dir, 'meta_controller')

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        policies = F.softmax(self.policy(x), dim=1)
        return policies
    
    def decide_policy(self, state):
        print("Original state:", state)  # Debug: Check what's being passed
        if not isinstance(state, T.Tensor):
            state = T.tensor(state, dtype=T.float).to(self.device)
        else:
            state = state.to(self.device)
        policies_prob = self.forward(state)
        # This can be a stochastic policy where you sample or you can take the argmax for deterministic behavior.
        policy_action = T.argmax(policies_prob, dim=1)
        print("Policy action:", policy_action)
        return policy_action
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def learn(self, state, reward, policy):
        self.optimizer.zero_grad()
        policies_prob = self.forward(state)
        log_prob = T.log(policies_prob[0, policy])
        loss = -log_prob * reward
        loss.backward()
        self.optimizer.step()
        return loss