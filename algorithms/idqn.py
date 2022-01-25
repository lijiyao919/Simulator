from collections import namedtuple
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from algorithms.models.mlp_net import MLP_Network
from simulator.timer import Timer
import numpy as np
import torch as T
import random
import math
import torch.nn as nn
from statistics import mean

# Device configuration
device = T.device('cuda' if T.cuda.is_available() else 'cpu')
print('The device is: ', device)

#Double DQN
DDQN = True

#Dueling DQN
DUELING = True

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

class IDQN_Agent(object):
    def __init__(self, input_dims, n_actions, fc1_dims, eta, buffer_size=10000, batch_size=32, gamma=0.99, target_update_feq=1000, eps_end=0.1, eps_decay=1000000):
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update_feq

        self.writer = SummaryWriter()
        self.policy_net = MLP_Network(input_dims, n_actions, fc1_dims, eta, self.writer).to(device)
        self.target_net = MLP_Network(input_dims, n_actions, fc1_dims, eta, self.writer).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.eps_start = 1
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.q_arr = deque(maxlen=10)

    def _get_next_state(self, driver):
        if driver.in_service:
            return np.array([Timer.get_time(driver.wake_up_time), Timer.get_day(driver.wake_up_time), driver.zid])
        else:
            return np.array([Timer.get_time(Timer.get_time_step()), Timer.get_day(Timer.get_time_step()), driver.zid])

    def store_exp(self, drivers, obs, actions, rewards, next_obs):
        time = Timer.get_time(Timer.get_time_step()-1)
        day = Timer.get_day(Timer.get_time_step()-1)
        next_time = Timer.get_time(Timer.get_time_step())
        next_day = Timer.get_day(Timer.get_time_step())
        assert 0 <= time <= 1440
        assert 1 <= day <= 7
        assert 0 <= next_time <= 1440
        assert 1 <= next_day <= 7

        for did, driver in drivers.items():
            A = actions[did]
            if A != -1:
                assert rewards[did] is not None
                state = np.array([time, day, obs["driver_locs"][did]])
                state_tensor = T.from_numpy(np.expand_dims(state.astype(np.float32), axis=0)).to(device)
                action_torch = T.tensor([[A]], device=device)
                reward = rewards[did]
                reward_torch = T.tensor([reward], device=device)
                next_state = self._get_next_state(driver)
                next_state_tensor = T.from_numpy(np.expand_dims(next_state.astype(np.float32), axis=0)).to(device)

                self.replay_buffer.push(state_tensor, action_torch, reward_torch, next_state_tensor)

    def select_action(self, obs, drivers, steps_done):
        actions = [-1] * len(drivers)
        time = Timer.get_time(Timer.get_time_step())
        day = Timer.get_day(Timer.get_time_step())
        assert 0 <= time <= 1440
        assert 1 <= day <= 7

        random_num = random.random()
        eps_thredhold = self.eps_end + (self.eps_start-self.eps_end)*math.exp(-1 * steps_done / self.eps_decay)
        for did, driver in drivers.items():
            if driver.on_line is True:
                assert obs["driver_locs"][did] == driver.zid
                if random_num > eps_thredhold:
                    with T.no_grad():
                        state = np.array([time, day, driver.zid])
                        state_tensor = T.from_numpy(np.expand_dims(state.astype(np.float32), axis=0)).to(device)
                        actions[did] = self.policy_net(state_tensor).max(1)[1].item()
                else:
                    actions[did] = random.randrange(self.n_actions)
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

        non_final_mask = T.tensor(tuple(map(lambda x: x is not None, batch.next_state)), device=device, dtype=T.bool)

        non_final_next_state_batch = T.cat([s for s in batch.next_state if s is not None]).to(device)

        #compute state action values
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_action_values = T.zeros(self.batch_size, device=device)
        if DDQN:
            max_act = self.policy_net(non_final_next_state_batch).max(1)[1].view(non_final_next_state_batch.size()[0], 1)
            next_state_action_values[non_final_mask] = self.target_net(non_final_next_state_batch).gather(1, max_act).detach().view(max_act.size()[0])
        else:
            next_state_action_values[non_final_mask] = self.target_net(non_final_next_state_batch).max(1)[0].detach()
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

        '''if step % log_feq*10 == 0:
            self.policy_net.save_checkpoint()'''

        #trace Q value and other parameters
        '''self.record_Q_value(state_action_values, step)
        if step % log_feq == 0:
            self.writer.add_scalar("Loss/train", loss, step)
            self.writer.add_scalar("The Average Q value", mean(self.q_arr), step)
            self.policy_net.traceWeight(step)
            self.policy_net.traceBias(step)
            self.policy_net.traceGrad(step)'''


    '''def record_Q_value(self, q_values, step):
        mean_q_value = T.mean(T.cat(tuple(q_values.detach()))).item()
        self.q_arr.append(mean_q_value)'''



    def flushTBSummary(self):
        self.writer.flush()

    def train_mode(self):
        self.policy_net.train()

    def test_mode(self):
        self.policy_net.eval()

