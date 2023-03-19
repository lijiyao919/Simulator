import numpy as np
import random
import math
import torch as T
from simulator.config import *
from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter
from algorithms.QLearners.models.mlp_net import MLP_Network
from algorithms.QLearners.agent import Agent
from algorithms.QLearners.agent import device
from simulator.timer import Timer
from collections import defaultdict


DDQN = False

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'success')) #Transition is a class, not object

class PrioritizedBuffer(object):
    def __init__(self, capacity, prob_alpha):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, success):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(state, action, reward, next_state, success))
        else:
            self.buffer[self.pos] = Transition(state, action, reward, next_state, success)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = Transition(*zip(*samples))
        states = np.concatenate(batch.state)
        actions = batch.action
        rewards = batch.reward
        next_states = np.concatenate(batch.next_state)
        successes = batch.success

        return states, actions, rewards, next_states, successes, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

class IPER_Agent(Agent):
    def __init__(self, input_dims, n_actions, fc1_dims, eta, buffer_size=1000, batch_size=128, gamma=0.99, target_update_feq=1000, \
                 alpha=0.6, beta_start=0.4, beta_frames = 10000, eps_end=0.1, eps_decay=25000):
        super(IPER_Agent, self).__init__()
        self.input_dims = input_dims
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

        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames

        self.replay_buffer = PrioritizedBuffer(buffer_size, self.alpha)

    def _softmax(self, x):
        return (np.exp(x) / np.exp(x).sum())

    def get_key(self, zid, act, reward):
        return str(zid) + ":" + str(act) + ":" + str(reward)

    def store_exp(self, drivers, obs, actions, rewards, next_obs):
        time = Timer.get_time(Timer.get_time_step() - 1)
        day = Timer.get_day(Timer.get_time_step() - 1)
        assert 0 <= time <= 1440
        assert 1 <= day <= 7

        finger_print_set = set()

        for did, driver in drivers.items():
            action = actions[did]
            if action != -1:
                assert rewards[did] is not None
                assert next_obs["driver_locs"][did] == driver.zid
                '''finger_print = self.get_key(driver.zid, action, rewards[did])
                if finger_print in finger_print_set:
                    continue
                finger_print_set.add(finger_print)'''
                state = IPER_Agent.get_state_dist_cmp(time, day, obs["driver_locs"][did], obs["on_call_rider_num"], obs["online_driver_num"])
                reward = rewards[did]
                next_state = self.get_next_state_dist_cmp(driver, next_obs["on_call_rider_num"], next_obs["online_driver_num"])
                if next_state is not None:
                    success = 0
                else:
                    success = 1
                    next_state = [0]*self.input_dims
                self.replay_buffer.push(state, action, reward, next_state, success)

    def select_action(self, obs, drivers, steps_done):
        actions = [-1] * len(drivers)
        time = Timer.get_time(Timer.get_time_step())
        day = Timer.get_day(Timer.get_time_step())
        assert 0 <= time <= 1440
        assert 1 <= day <= 7

        cache = defaultdict(lambda: None)
        random_num = random.random()
        eps_thredhold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1 * steps_done / self.eps_decay)
        for did, driver in drivers.items():
            if driver.on_line is True:
                assert obs["driver_locs"][did] == driver.zid
                if random_num > eps_thredhold:
                    if cache[driver.zid] is not None:
                        actions[did] = cache[driver.zid]
                    else:
                        with T.no_grad():
                            state = IPER_Agent.get_state_dist_cmp(time, day, driver.zid, obs["on_call_rider_num"],obs["online_driver_num"])
                            state_tensor = T.from_numpy(np.expand_dims(state.astype(np.float32), axis=0)).to(device)
                            scores = self.policy_net(state_tensor).cpu().numpy()[0]
                            probs = self._softmax(scores)
                            actions[did] = np.random.choice(N_ACTIONS, replace=False, p=probs)
                else:
                    actions[did] = random.randrange(self.n_actions)
        return actions

    def update(self, step):
        beta_by_frame = min(1.0, self.beta_start + step * (1.0 - self.beta_start) / self.beta_frames)
        state, action, reward, next_state, done, indices, weights = self.replay_buffer.sample(self.batch_size, beta_by_frame)

        state = T.tensor(state, dtype=T.float32, device=device)
        action = T.tensor(action, dtype=T.long, device=device)
        reward = T.tensor(reward, dtype=T.float32, device=device)
        next_state = T.tensor(next_state, dtype=T.float32, device=device)
        done = T.tensor(done, dtype=T.long, device=device)
        weights = T.tensor(weights, dtype=T.float32, device=device)

        q_values = self.policy_net(state)
        next_q_values = self.target_net(next_state)

        if DDQN:
            max_q_act = q_values.max(1)[1]
            next_q_value = next_q_values.gather(1,max_q_act.unsqueeze(1)).squeeze(1)
        else:
            next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        loss = (q_value - expected_q_value.detach()).pow(2) * weights
        prios = loss + 1e-5
        loss = loss.mean()

        self.policy_net.optimizer.zero_grad()
        loss.backward()
        self.replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
        self.policy_net.optimizer.step()

        if step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        #if step % log_feq == 0:
        #    self.writer.add_scalar("Loss/train", loss, step)

    def calcPerformance(self, aver_reward, step):
        self.writer.add_scalar("The Average Reward (10 episodes)", aver_reward, step)
        print('Step {}\tThe Average Reward (10 episodes): {:.2f}'.format(step, aver_reward))

    def train_mode(self):
        self.policy_net.train()

    def test_mode(self):
        self.policy_net.eval()

    def flushTBSummary(self):
        self.writer.flush()

