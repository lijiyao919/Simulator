import math
from simulator.config import *
from algorithms.QLearners.idqn import IDQN_Agent
from algorithms.QLearners.idqn import device
from simulator.timer import Timer
from data.graph import AdjList_Chicago
import torch as T
import numpy as np
import random


class AM_DQN_Agent(IDQN_Agent):
    def __init__(self, input_dims, n_actions, fc1_dims, eta, buffer_size=1000, batch_size=128, gamma=0.99, target_update_feq=1000, eps_end=0.1, eps_decay=25000):
        super().__init__(input_dims, n_actions, fc1_dims, eta, buffer_size, batch_size, gamma, target_update_feq, eps_end, eps_decay)

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
                adj_num = self.get_adj_zone_num(driver.zid)
                if random_num > eps_thredhold:
                    with T.no_grad():
                        state = AM_DQN_Agent.get_state_dist_cmp(time, day, driver.zid, obs["on_call_rider_num"], obs["online_driver_num"])
                        state_tensor = T.from_numpy(np.expand_dims(state.astype(np.float32), axis=0)).to(device)
                        scores = self.policy_net(state_tensor).cpu().numpy()[0][0:adj_num+1]
                        probs = self.softmax(scores)
                        actions[did] = np.random.choice(adj_num+1, replace=False, p=probs)
                else:
                    actions[did] = random.randrange(N_ACTIONS)
        return actions

