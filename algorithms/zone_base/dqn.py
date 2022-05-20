from algorithms.zone_base.agent import device
from algorithms.zone_base.agent import Agent, N_ACTIONS
from collections import namedtuple
from collections import deque
from data.graph import AdjList_Chicago
from simulator.timer import Timer
from simulator.config import *
from torch.utils.tensorboard import SummaryWriter
from algorithms.models.mlp_net import MLP_Network
import random
import torch.nn.functional as F
import torch as T
import numpy as np
import torch.nn as nn
from torch.distributions import Categorical
import math

SAMPLE="softmax"

#Double DQN
DDQN = False

#Load checkpoints file
LOAD = False

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state')) #Transition is a class, not object

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque([], maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQN_Agent(Agent):
    def __init__(self, input_dims, n_actions, fc1_dims, eta, buffer_size=10000, batch_size=32, gamma=0.99, target_update_feq=1000, eps_end=0.1, eps_decay=1000000):
        super(DQN_Agent, self).__init__()
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update_feq

        self.writer = SummaryWriter()
        self.policy_net = MLP_Network(input_dims, n_actions, fc1_dims, eta, self.writer, chkpt_file='dqn_nwk.pth').to(device)
        self.target_net = MLP_Network(input_dims, n_actions, fc1_dims, eta, self.writer).to(device)

        if LOAD:
            print("Load from: dqn_nwk.pth")
            self.policy_net.load_checkpoint()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.eps_start = 1
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.q_arr = deque(maxlen=10)

    def store_exp(self, obs, rewards, next_obs):
        for zid in range(1, TOTAL_ZONES+1):
            state = DQN_Agent.get_state_local_dist(zid, obs["on_call_rider_num"], obs["online_driver_num"])
            state_tensor = T.from_numpy(np.expand_dims(state.astype(np.float32), axis=0)).to(device)
            next_state = DQN_Agent.get_state_local_dist(zid, next_obs["on_call_rider_num"], next_obs["online_driver_num"])
            next_state_tensor = T.from_numpy(np.expand_dims(next_state.astype(np.float32), axis=0)).to(device)
            for act in range(N_ACTIONS):
                action_torch = T.tensor([[act]], device=device)
                reward = rewards[zid][act]
                reward_torch = T.tensor([reward], device=device)
            self.replay_buffer.push(state_tensor, action_torch, reward_torch, next_state_tensor)

    def select_actions(self, drivers, obs):
        act_dists = {}
        actions = [-1] * len(drivers)

        #calculate the action distributuion of each zone
        for zid in range(1, TOTAL_ZONES+1):
                with T.no_grad():
                    state = DQN_Agent.get_state_local_dist(zid, obs["on_call_rider_num"], obs["online_driver_num"])
                    state_tensor = T.from_numpy(np.expand_dims(state.astype(np.float32), axis=0)).to(device)
                    act_dists[zid] = self.policy_net(state_tensor)

        #select action for each driver
        for did, driver in drivers.items():
            if driver.on_line is True:
                act_dist = act_dists[driver.zid]
                probs = F.softmax(act_dist, dim=1)
                m = Categorical(probs)
                action = m.sample()
                actions[did] = action.item()
        return actions

    def update(self, step):
        if len(self.replay_buffer) < self.batch_size:
            return

        #handling samples from
        samples = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*samples))
        state_batch = T.cat(batch.state).to(device)
        action_batch = T.cat(batch.action)
        reward_batch = T.cat(batch.reward)
        next_state_batch = T.cat(batch.next_state).to(device)

        #compute state action values
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        #target Q value
        next_state_action_values = self.target_net(next_state_batch).max(1)[0].detach()
        target_state_action_values = next_state_action_values*self.gamma+reward_batch


        #compute huber loss and optimize
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, target_state_action_values.unsqueeze(1))
        self.policy_net.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.policy_net.optimizer.step()
       
        if step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


    def flushTBSummary(self):
        self.writer.flush()

    def train_mode(self):
        self.policy_net.train()

    def test_mode(self):
        self.policy_net.eval()