from algorithms.PGs.models.mlp_net import MLP_Net
import torch as T
import numpy as np
from collections import namedtuple, defaultdict
from torch.distributions import Categorical
from algorithms.agent import Agent
from algorithms.agent import device
from simulator.timer import Timer

Transition = namedtuple('Transition', ('state', 'action', 'log_prob', 'value', 'reward', 'success', 'next_state')) #Transition is a class, not object

class RolloutStorage(object):
    def __init__(self):
        self.roll_out = []

    def push(self, state, action, log_prob, value, reward, success, next_state):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.roll_out.append(Transition(state, action, log_prob, value, reward, success, next_state))

    def sample(self, batch_size):
        samples = [self.roll_out[idx] for idx in range(len(self.roll_out))]

        batch = Transition(*zip(*samples))
        states = T.tensor(np.concatenate(batch.state), dtype=T.float32, device=device)
        actions = T.cat(batch.action).to(device)
        old_log_probs = T.cat(batch.log_prob).to(device).detach()
        old_values = T.cat(batch.value).to(device).detach()
        rewards = T.tensor(batch.reward, dtype=T.float32, device=device)
        successes = T.tensor(batch.success, dtype=T.long, device=device)
        next_states = T.tensor(np.concatenate(batch.next_state), dtype=T.float32, device=device)

        return states, actions, old_log_probs, old_values, rewards, successes, next_states

    def clear(self):
        del self.roll_out[:]


class PPO_Agent(Agent):
    def __init__(self, input_dims, n_actions, fc1_dims, fc2_dims, eta, gamma=0.99, batch_size=32, ppo_step=4, clip_param=0.2):
        super(PPO_Agent, self).__init__()
        self.input_dims = input_dims
        self.memo = RolloutStorage()
        self.gamma = gamma
        self.batch_size = batch_size
        self.policy_net = MLP_Net(input_dims, n_actions, fc1_dims, fc2_dims, eta).to(device)
        self.ppo_step = ppo_step
        self.clip_param = clip_param

    def store_exp(self, drivers, actions, log_probs, values, rewards, next_obs):
        time = Timer.get_time(Timer.get_time_step() - 1)
        day = Timer.get_day(Timer.get_time_step() - 1)
        assert 0 <= time <= 1440
        assert 1 <= day <= 7

        for did, driver in drivers.items():
            if actions[did] != -1:
                assert log_probs[did] is not None
                assert values[did] is not None
                assert rewards[did] is not None
                state = PPO_Agent.get_state(time, day, driver.zid)
                next_state = PPO_Agent.get_next_state(driver)
                if next_state is not None:
                    success = 0
                else:
                    success = 1
                    next_state = [0] * self.input_dims
                self.memo.push(state, actions[did], log_probs[did], values[did], rewards[did], success, next_state)


    def feed_forward(self, obs, drivers):
        actions = [-1] * len(drivers)
        log_probs = [None] * len(drivers)
        values = [None] * len(drivers)
        time = Timer.get_time(Timer.get_time_step())
        day = Timer.get_day(Timer.get_time_step())
        assert 0 <= time <= 1440
        assert 1 <= day <= 7

        cache = defaultdict(lambda: None)
        for did, driver in drivers.items():
            if driver.on_line is True:
                assert obs["driver_locs"][did] == driver.zid
                if cache[driver.zid] is None:
                    state = PPO_Agent.get_state(time, day, driver.zid)
                    state_tensor = T.from_numpy(np.expand_dims(state.astype(np.float32), axis=0)).to(device)
                    prob, value = self.policy_net(state_tensor)
                    dist = Categorical(prob)
                    cache[driver.zid] = (dist, value)
                else:
                    dist, value = cache[driver.zid]
                action = dist.sample()
                log_prob = dist.log_prob(action)
                actions[did] = action
                log_probs[did] = log_prob
                values[did] = value[0]
        return actions, log_probs, values

    def learn(self):
        state_tensor, actions_tensor, old_log_probs_tensor, old_values_tensor, rewards_tensor, successes_tensor, next_states_tensor = \
            self.memo.sample(self.batch_size)

        with T.no_grad():
            _, next_v = self.policy_net(next_states_tensor)
        next_v = next_v.view(next_v.size(dim=0), )
        target_values_tensor = rewards_tensor + self.gamma * next_v
        advantages = target_values_tensor - old_values_tensor
        #print(advantage)

        for _ in range(self.ppo_step):
            new_prob, new_values = self.policy_net(state_tensor)
            dist = Categorical(new_prob)
            new_log_prob = dist.log_prob(actions_tensor)
            new_entropy = dist.entropy().mean()
            new_values = new_values.squeeze(1)

            ratio = (new_log_prob - old_log_probs_tensor).exp()
            surr1 = ratio * advantages
            surr2 = T.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages

            # compute loss
            actor_loss = - T.min(surr1, surr2).mean()
            critic_loss = (target_values_tensor - new_values).pow(2).mean()
            loss = actor_loss + 0.5 * critic_loss - 0.001 * new_entropy

            self.policy_net.optimizer.zero_grad()
            loss.backward()
            T.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.policy_net.optimizer.step()

        self.memo.clear()

    def train_mode(self):
        self.policy_net.train()

    def test_mode(self):
        self.policy_net.eval()