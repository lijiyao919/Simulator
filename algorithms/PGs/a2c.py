from algorithms.PGs.models.mlp_net import MLP_Net
from algorithms.PGs.models.mlp_dist_net import DIST_Net
import torch as T
import numpy as np
from collections import namedtuple, defaultdict
from torch.distributions import Categorical
from algorithms.agent import Agent
from algorithms.agent import device
from simulator.timer import Timer
from simulator.config import *

LOAD = False

Transition = namedtuple('Transition', ('log_prob', 'value', 'reward', 'success', 'next_s',  'next_d', 'next_z', 'entropy')) #Transition is a class, not object

class RolloutStorage(object):
    def __init__(self):
        self.roll_out = []

    def push(self, log_prob, value, reward, success, next_s, next_d, next_z, entropy):
        #next_state = np.expand_dims(next_state, 0)
        self.roll_out.append(Transition(log_prob, value, reward, success, next_s, next_d, next_z, entropy))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.roll_out), batch_size)
        samples = [self.roll_out[idx] for idx in indices]

        batch = Transition(*zip(*samples))
        log_probs = T.cat(batch.log_prob).to(device)
        values = T.cat(batch.value).to(device)
        rewards = T.tensor(batch.reward, dtype=T.float32, device=device)
        successes = T.tensor(batch.success, dtype=T.long, device=device)
        #next_states = T.tensor(np.concatenate(batch.next_state), dtype=T.float32, device=device)
        next_s = T.tensor(np.stack(batch.next_s), dtype=T.float32, device=device)
        next_d = T.tensor(np.stack(batch.next_d), dtype=T.float32, device=device)
        next_z = T.tensor(np.stack(batch.next_z), dtype=T.float32, device=device)
        entropy = T.stack(batch.entropy).to(device).sum()

        return log_probs, values, rewards, successes, next_s, next_d, next_z, entropy

    def clear(self):
        del self.roll_out[:]


class A2C_Agent(Agent):
    def __init__(self, dist_dims, n_actions, gamma=0.99, batch_size=32):
        super(A2C_Agent, self).__init__()
        self.dist_dims = dist_dims
        self.memo = RolloutStorage()
        self.gamma = gamma
        self.batch_size = batch_size
        self.policy_net = DIST_Net(dist_dims, n_actions, 64, 32, 256, 128, 0.001).to(device)

        if LOAD:
            print("Load from: a2c.pth")
            self.policy_net.load_checkpoint()


    def store_exp(self, drivers, log_probs, values, rewards, next_obs, entropys, actions):
        time = Timer.get_time(Timer.get_time_step() - 1)
        day = Timer.get_day(Timer.get_time_step() - 1)
        assert 0 <= time <= 1440
        assert 1 <= day <= 7

        for did, driver in drivers.items():
            if actions[did] != -1:
                assert log_probs[did] is not None
                assert values[did] is not None
                assert rewards[did] is not None
                assert entropys[did] is not None
                #next_state = A2C_Agent.get_next_state(driver)
                s_dist, d_dist, z_code = A2C_Agent.get_next_state_dist(driver, next_obs["on_call_rider_num"], next_obs["online_driver_num"])
                if s_dist is not None:
                    success = 0
                else:
                    success = 1
                    s_dist = [0] * 77
                    d_dist = [0] * 77
                    z_code = [0] * 77
                self.memo.push(log_probs[did], values[did], rewards[did], success, s_dist, d_dist, z_code, entropys[did])

    def read_V(self, obs=None):
        time = Timer.get_time(Timer.get_time_step())
        day = Timer.get_day(Timer.get_time_step())
        assert 0 <= time <= 1440
        assert 1 <= day <= 7
        V = defaultdict(lambda: None)  # zid:value

        s_dist, d_dist = A2C_Agent.get_state_dist(obs["on_call_rider_num"], obs["online_driver_num"])
        d_dist = T.from_numpy(np.expand_dims(d_dist.astype(np.float32), axis=0)).to(device)
        s_dist = T.from_numpy(np.expand_dims(s_dist.astype(np.float32), axis=0)).to(device)

        for zid in range(1, TOTAL_ZONES+1):
            z_code = A2C_Agent.get_state_zid(zid)
            z_code = T.from_numpy(np.expand_dims(z_code.astype(np.float32), axis=0)).to(device)
            #state = A2C_Agent.get_state(time, day, zid)
            #state_tensor = T.from_numpy(np.expand_dims(state.astype(np.float32), axis=0)).to(device)
            with T.no_grad():
                _, value = self.policy_net(s_dist, d_dist, z_code)
            V[zid] = value[0].item()
        return V


    def feed_forward(self, obs, drivers):
        actions = [-1] * len(drivers)
        log_probs = [None] * len(drivers)
        values = [None] * len(drivers)
        dist_entropys = [None] * len(drivers)
        time = Timer.get_time(Timer.get_time_step())
        day = Timer.get_day(Timer.get_time_step())
        assert 0 <= time <= 1440
        assert 1 <= day <= 7

        cache = defaultdict(lambda: None)
        s_dist, d_dist = A2C_Agent.get_state_dist(obs["on_call_rider_num"], obs["online_driver_num"])
        d_dist = T.from_numpy(np.expand_dims(d_dist.astype(np.float32), axis=0)).to(device)
        s_dist = T.from_numpy(np.expand_dims(s_dist.astype(np.float32), axis=0)).to(device)
        for did, driver in drivers.items():
            if driver.on_line is True:
                assert obs["driver_locs"][did] == driver.zid
                if cache[driver.zid] is None:
                    #state = A2C_Agent.get_state(time, day, driver.zid)
                    #state_tensor = T.from_numpy(np.expand_dims(state.astype(np.float32), axis=0)).to(device)
                    z_code = A2C_Agent.get_state_zid(driver.zid)
                    z_code = T.from_numpy(np.expand_dims(z_code.astype(np.float32), axis=0)).to(device)
                    #prob, value = self.policy_net(state_tensor)
                    prob, value = self.policy_net(s_dist, d_dist, z_code)
                    dist = Categorical(prob)
                    cache[driver.zid] = (dist, value)
                else:
                    dist, value = cache[driver.zid]
                action = dist.sample()
                log_prob = dist.log_prob(action)
                actions[did] = action
                log_probs[did] = log_prob
                values[did] = value[0]
                dist_entropys[did] = dist.entropy().mean()
        return actions, log_probs, values, dist_entropys

    def learn(self):
        log_probs_tensor, values_tensor, rewards_tensor, successes_tensor, next_s, next_d, next_z, total_entropy_tensor = self.memo.sample(self.batch_size)

        with T.no_grad():
            #_, next_v = self.policy_net(next_states_tensor)
            _, next_v = self.policy_net(next_s, next_d, next_z)
        next_v = next_v.view(next_v.size(dim=0), )
        target_values_tensor = rewards_tensor + self.gamma * next_v
        advantage = target_values_tensor - values_tensor
        #print(advantage)

        #compute loss
        actor_loss = (-log_probs_tensor*advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + 0.5*critic_loss - 0.001*total_entropy_tensor

        self.policy_net.optimizer.zero_grad()
        loss.backward()
        T.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.policy_net.optimizer.step()

        self.memo.clear()

    def train_mode(self):
        self.policy_net.train()

    def test_mode(self):
        self.policy_net.eval()