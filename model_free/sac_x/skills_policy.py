import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SkillPolicy(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims=400, fc2_dims=300, name='skill_policy', chkpt_dir='tmp/sac_x'):
        super(SkillPolicy, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0003)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.name)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = T.tanh(self.fc3(x))
        return actions

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def compute_intrinsic_reward(self, state, action, new_state, done):
        # Example: Reward is simply the mean of the squared actions, encouraging smaller action values
        return -T.mean(action**2).item()

class AvoidObstaclesPolicy(nn.Module):
    def __init__(self, input_dims, fc1_dims=400, fc2_dims=300, chkpt_dir='tmp/sac_x'):
        super(AvoidObstaclesPolicy, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.action = nn.Linear(fc2_dims, 1)  # Assume single action output for simplicity
        self.optimizer = optim.Adam(self.parameters(), lr=0.0003)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.checkpoint_file = os.path.join(chkpt_dir, 'avoid_obstacles_checkpoint')

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = T.sigmoid(self.action(x))  # Using sigmoid for a bounded action space
        return action
    
    def compute_intrinsic_reward(self, state, action, new_state, done):
        # Assuming smaller action values mean closer to obstacles
        return -1.0 / (action + 1e-5)  # Adding a small constant to avoid division by zero

class TargetDistancePolicy(nn.Module):
    def __init__(self, input_dims, target_position, optimal_distance=1.0, fc1_dims=400, fc2_dims=300, chkpt_dir='tmp/sac_x'):
        super(TargetDistancePolicy, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, 1)  # Outputting a single value representing the action
        self.optimizer = optim.Adam(self.parameters(), lr=0.0003)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.target_position = T.tensor(target_position, dtype=T.float).to(self.device)
        self.optimal_distance = optimal_distance
        self.checkpoint_file = os.path.join(chkpt_dir, 'target_distance_checkpoint')

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = T.tanh(self.fc3(x))  # Assume some action space normalization
        return action

    def compute_intrinsic_reward(self, current_position):
        """ Calculate the reward based on the current position of the agent relative to the target. """
        current_distance = T.norm(current_position - self.target_position)
        reward = -T.abs(current_distance - self.optimal_distance)
        return reward
