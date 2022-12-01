import os
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DIST_Net(nn.Module):
    def __init__(self, dist_dims, n_actions, fce1_dims, fce2_dims, fc1_dims, fc2_dims, eta, chkpt_dir='../checkpoints', chkpt_file='a2c.pth'):
        super(DIST_Net, self).__init__()

        self.fc_supply_1 = nn.Linear(dist_dims, fce1_dims)
        self.fc_demand_1 = nn.Linear(dist_dims, fce1_dims)

        self.fc_supply_2 = nn.Linear(fce1_dims, fce2_dims)
        self.fc_demand_2 = nn.Linear(fce1_dims, fce2_dims)

        self.fc1 = nn.Linear(2*fce2_dims+dist_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.v = nn.Linear(fc2_dims, 1)
        self.pi = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=eta)  # 0.0001

        self.checkpoint_file = os.path.join(chkpt_dir, chkpt_file)

    def forward(self, supply_dist, demand_dist, zid):
        encode_supply = F.relu(self.fc_supply_1(supply_dist))
        encode_demand = F.relu(self.fc_demand_1(demand_dist))
        encode_supply = F.relu(self.fc_supply_2(encode_supply))
        encode_demand = F.relu(self.fc_demand_2(encode_demand))
        encode_out = T.cat((encode_supply, encode_demand, zid), dim=1)
        x = F.relu(self.fc1(encode_out))
        x = F.relu(self.fc2(x))
        pi = self.pi(x)
        v = self.v(x)
        return (F.softmax(pi, dim=1), v)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))