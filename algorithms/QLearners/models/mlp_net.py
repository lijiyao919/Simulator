import os
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

DUELING = True

class MLP_Network(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims, eta, tb_writer, chkpt_dir='../checkpoints', chkpt_file='mlp_nwk.pth'):
        super(MLP_Network, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc1_pi = nn.Linear(input_dims, fc1_dims)
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()
        self.fc3 = nn.Linear(fc1_dims, n_actions)
        self.fc2 = nn.Linear(fc1_dims, 1)

        self.optimizer = optim.RMSprop(self.parameters(), lr=eta)    #0.0001
        # self.optimizer = optim.SGD(self.__policy.parameters(), lr=0.0001)  # 0.01 for method2, 3, online learn
        self.optimizer.zero_grad()

        self.checkpoint_file = os.path.join(chkpt_dir, chkpt_file)
        self.writer = tb_writer

    def forward(self, x):
        if DUELING:
            x_a = self.tanh(self.fc1(x))
            x_b = self.tanh(self.fc1_pi(x))
            V = self.fc2(x_a)
            A = self.fc3(x_b)
            AVER_A = T.mean(A, dim=1, keepdim=True)
            return V + (A - AVER_A)
        else:
            x = self.fc1(x)
            #x = self.elu(x)
            x = self.tanh(x)
            action_scores = self.fc3(x)
            return action_scores

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    '''def traceWeight(self, epoch):
        self.writer.add_histogram('fc1.weight', self.fc1.weight, epoch)
        self.writer.add_histogram('fc3.weight', self.fc3.weight, epoch)

    def traceBias(self, epoch):
        self.writer.add_histogram('fc1.bias', self.fc1.bias, epoch)
        self.writer.add_histogram('fc3.bias', self.fc3.bias, epoch)

    def traceGrad(self, epoch):
        self.writer.add_histogram('fc1.weight.grad', self.fc1.weight.grad, epoch)
        self.writer.add_histogram('fc1.bias.grad', self.fc1.bias.grad, epoch)
        self.writer.add_histogram('fc3.weight.grad', self.fc3.weight.grad, epoch)
        self.writer.add_histogram('fc3.bias.grad', self.fc3.bias.grad, epoch)'''