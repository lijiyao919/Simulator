from algorithms.PGs.models.mlp_net import MLP_Net
from algorithms.PGs.models.mlp_dist_net import DIST_Net
from algorithms.PGs.models.rnn_dist_net import RNN_DIST_Net
import torch as T
import numpy as np
from collections import namedtuple, defaultdict
from torch.distributions import Categorical
from algorithms.agent import Agent
from algorithms.agent import device
from simulator.timer import Timer
from simulator.config import *
from data.graph import AdjList_Chicago

LOAD = False

Transition = namedtuple('Transition', ('log_prob', 'value', 'reward', 'success', 'next_s',  'next_d', 'next_z', 'entropy', 'hidden_s', 'hidden_d')) #Transition is a class, not object

class RolloutStorage(object):
    def __init__(self):
        self.roll_out = []

    def push(self, log_prob, value, reward, success, next_s, next_d, next_z, entropy, hidden_s, hidden_d):
        #next_state = np.expand_dims(next_state, 0)
        self.roll_out.append(Transition(log_prob, value, reward, success, next_s, next_d, next_z, entropy, hidden_s, hidden_d))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.roll_out), batch_size)
        samples = [self.roll_out[idx] for idx in indices]

        batch = Transition(*zip(*samples))
        log_probs = T.cat(batch.log_prob).to(device)
        values = T.cat(batch.value).to(device)
        rewards = T.tensor(batch.reward, dtype=T.float32, device=device)
        successes = T.tensor(batch.success, dtype=T.long, device=device)
        #next_states = T.tensor(np.concatenate(batch.next_state), dtype=T.float32, device=device)
        next_s = T.tensor(np.stack(batch.next_s), dtype=T.float32, device=device).unsqueeze(0)
        next_d = T.tensor(np.stack(batch.next_d), dtype=T.float32, device=device).unsqueeze(0)
        next_z = T.tensor(np.stack(batch.next_z), dtype=T.float32, device=device)
        hidden_s = T.cat(batch.hidden_s, dim=1)
        hidden_d = T.cat(batch.hidden_d, dim=1)
        entropy = T.stack(batch.entropy).to(device).sum()

        return log_probs, values, rewards, successes, next_s, next_d, next_z, entropy, hidden_s, hidden_d

    def clear(self):
        del self.roll_out[:]


class A2C_Agent(Agent):
    def __init__(self, dist_dims, n_actions, gamma=0.99, batch_size=32):
        super(A2C_Agent, self).__init__()
        self.dist_dims = dist_dims
        self.memo = RolloutStorage()
        self.gamma = gamma
        self.batch_size = batch_size
        #self.policy_net = DIST_Net(dist_dims, n_actions, 64, 32, 256, 128, 0.001).to(device)
        self.policy_net = RNN_DIST_Net(dist_dims, n_actions, 32, 1, 256, 128, 0.0005).to(device)
        self.hidden_supply = defaultdict(lambda : None)
        self.hidden_demand = defaultdict(lambda : None)

        if LOAD:
            print("Load from: a2c.pth")
            self.policy_net.load_checkpoint()

    def reset_hidden(self):
        for zid in range(1, TOTAL_ZONES + 1):
            self.hidden_supply[zid] = self.policy_net.initHidden()
            self.hidden_demand[zid] = self.policy_net.initHidden()

    def store_exp(self, drivers, log_probs, values, rewards, next_obs, entropys, actions, hidden_s, hidden_d):
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
                assert hidden_s[did] is not None
                assert hidden_d[did] is not None
                #next_state = A2C_Agent.get_next_state(driver)
                s_dist, d_dist, z_code = A2C_Agent.get_next_state_dist(driver, next_obs["on_call_rider_num"], next_obs["online_driver_num"])
                if s_dist is not None:
                    success = 0
                else:
                    success = 1
                    s_dist = [0] * 77
                    d_dist = [0] * 77
                    z_code = [0] * 77
                self.memo.push(log_probs[did], values[did], rewards[did], success, s_dist, d_dist, z_code, entropys[did], hidden_s[did], hidden_d[did])

    def read_V(self, obs=None):
        time = Timer.get_time(Timer.get_time_step())
        day = Timer.get_day(Timer.get_time_step())
        assert 0 <= time <= 1440
        assert 1 <= day <= 7
        V = defaultdict(lambda: None)  # zid:value

        s_dist, d_dist = A2C_Agent.get_state_dist(obs["on_call_rider_num"], obs["online_driver_num"])
        d_dist = T.from_numpy(np.expand_dims(d_dist.astype(np.float32), axis=0)).to(device)
        s_dist = T.from_numpy(np.expand_dims(s_dist.astype(np.float32), axis=0)).to(device)

        d_dist = d_dist.unsqueeze(0)
        s_dist = s_dist.unsqueeze(0)

        for zid in range(1, TOTAL_ZONES+1):
            z_code = A2C_Agent.get_state_zid(zid)
            z_code = T.from_numpy(np.expand_dims(z_code.astype(np.float32), axis=0)).to(device)
            #state = A2C_Agent.get_state(time, day, zid)
            #state_tensor = T.from_numpy(np.expand_dims(state.astype(np.float32), axis=0)).to(device)
            with T.no_grad():
                #_, value = self.policy_net(s_dist, d_dist, z_code)
                _, value, h_s, h_d = self.policy_net(s_dist, d_dist, z_code, self.hidden_supply[zid], self.hidden_demand[zid])
            V[zid] = value[0].item()
            self.hidden_supply[zid] = h_s
            self.hidden_demand[zid] = h_d
        return V


    def feed_forward(self, obs, drivers):
        actions = [-1] * len(drivers)
        log_probs = [None] * len(drivers)
        values = [None] * len(drivers)
        dist_entropys = [None] * len(drivers)
        hiddens_supply = [None] * len(drivers)
        hiddens_demand = [None] * len(drivers)
        time = Timer.get_time(Timer.get_time_step())
        day = Timer.get_day(Timer.get_time_step())
        assert 0 <= time <= 1440
        assert 1 <= day <= 7

        cache = defaultdict(lambda: None)
        s_dist, d_dist = A2C_Agent.get_state_dist(obs["on_call_rider_num"], obs["online_driver_num"])
        d_dist = T.from_numpy(np.expand_dims(d_dist.astype(np.float32), axis=0)).to(device)
        s_dist = T.from_numpy(np.expand_dims(s_dist.astype(np.float32), axis=0)).to(device)

        d_dist = d_dist.unsqueeze(0)
        s_dist = s_dist.unsqueeze(0)

        for did, driver in drivers.items():
            if driver.on_line is True:
                assert obs["driver_locs"][did] == driver.zid
                if cache[driver.zid] is None:
                    #state = A2C_Agent.get_state(time, day, driver.zid)
                    #state_tensor = T.from_numpy(np.expand_dims(state.astype(np.float32), axis=0)).to(device)
                    z_code = A2C_Agent.get_state_zid(driver.zid)
                    z_code = T.from_numpy(np.expand_dims(z_code.astype(np.float32), axis=0)).to(device)
                    #prob, value = self.policy_net(state_tensor)
                    prob, value, h_s, h_d = self.policy_net(s_dist, d_dist, z_code, self.hidden_supply[driver.zid], self.hidden_demand[driver.zid])
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
                hiddens_supply[did] = h_s
                hiddens_demand[did] = h_d
        return actions, log_probs, values, dist_entropys, hiddens_supply, hiddens_demand

    def fake_exp_maker(self, obs, lost_locs):
        cache = defaultdict(lambda: None)
        s_dist, d_dist = A2C_Agent.get_state_dist(obs["on_call_rider_num"], obs["online_driver_num"])
        d_dist = T.from_numpy(np.expand_dims(d_dist.astype(np.float32), axis=0)).to(device)
        s_dist = T.from_numpy(np.expand_dims(s_dist.astype(np.float32), axis=0)).to(device)
        d_dist = d_dist.unsqueeze(0)
        s_dist = s_dist.unsqueeze(0)

        for zid in lost_locs:

            #lost loc
            if cache[zid] is None:
                z_code = A2C_Agent.get_state_zid(zid)
                z_code = T.from_numpy(np.expand_dims(z_code.astype(np.float32), axis=0)).to(device)
                prob, value, h_s, h_d = self.policy_net(s_dist, d_dist, z_code, self.hidden_supply[zid], self.hidden_demand[zid])
                dist = Categorical(prob)
                cache[zid] = (dist, value)
            else:
                dist, value = cache[zid]
            action = T.tensor(len(AdjList_Chicago[zid]), device=device)
            log_prob = dist.log_prob(action)
            entropy = dist.entropy().mean()
            self.memo.push(log_prob, value[0], 1, 1, [0] * 77, [0] * 77, [0] * 77, entropy, h_s, h_d)


            for adj_zid in AdjList_Chicago[zid]:
                if cache[adj_zid] is None:
                    z_code = A2C_Agent.get_state_zid(adj_zid)
                    z_code = T.from_numpy(np.expand_dims(z_code.astype(np.float32), axis=0)).to(device)
                    prob, value, h_s, h_d = self.policy_net(s_dist, d_dist, z_code, self.hidden_supply[adj_zid], self.hidden_demand[adj_zid])
                    dist = Categorical(prob)
                    cache[adj_zid] = (dist, value)
                else:
                    dist, value = cache[adj_zid]
                if zid not in AdjList_Chicago[adj_zid]:
                    #print(AdjList_Chicago[adj_zid], adj_zid, zid)
                    continue
                idx = AdjList_Chicago[adj_zid].index(zid)
                action = T.tensor(idx, device=device)
                log_prob = dist.log_prob(action)
                entropy = dist.entropy().mean()
                self.memo.push(log_prob, value[0], 1, 1, [0]*77, [0]*77, [0]*77, entropy, h_s, h_d)

    def learn(self):
        log_probs_tensor, values_tensor, rewards_tensor, successes_tensor, next_s, next_d, next_z, total_entropy_tensor, hidden_s, hidden_d = \
            self.memo.sample(self.batch_size)

        with T.no_grad():
            _, next_v, _, _ = self.policy_net(next_s, next_d, next_z, hidden_s, hidden_d)
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