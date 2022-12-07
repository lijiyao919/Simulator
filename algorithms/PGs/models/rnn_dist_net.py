import os
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from algorithms.agent import device

class RNN_DIST_Net(nn.Module):
    def __init__(self, dist_dims, n_actions, n_hidden, n_layers, fc1_dims, fc2_dims, eta, chkpt_dir='../checkpoints', chkpt_file='a2c.pth'):
        super(RNN_DIST_Net, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        self.rnn_s = nn.GRU(77, n_hidden, n_layers)
        self.rnn_d = nn.GRU(77, n_hidden, n_layers)

        self.fc_s = nn.Linear(n_hidden, 32)
        self.fc_d = nn.Linear(n_hidden, 32)

        self.fc1 = nn.Linear(2*32+dist_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.v = nn.Linear(fc2_dims, 1)
        self.pi = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=eta)  # 0.0001

        self.checkpoint_file = os.path.join(chkpt_dir, chkpt_file)

    def forward(self, supply_dist, demand_dist, zid, hidden_in_s, hidden_in_d):
        out_s, hidden_out_s = self.rnn_s(supply_dist, hidden_in_s)
        out_s = out_s[-1, :, :]

        out_d, hidden_out_d = self.rnn_d(demand_dist, hidden_in_d)
        out_d = out_d[-1, :, :]

        encode_supply = self.fc_s(out_s)
        encode_demand = self.fc_d(out_d)
        encode_out = T.cat((encode_supply, encode_demand, zid), dim=1)
        x = F.relu(self.fc1(encode_out))
        x = F.relu(self.fc2(x))
        pi = self.pi(x)
        v = self.v(x)
        return (F.softmax(pi, dim=1), v, hidden_out_s, hidden_out_d)

    def initHidden(self):
        return T.zeros(self.n_layers, 1, self.n_hidden, device=device)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))