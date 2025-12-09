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