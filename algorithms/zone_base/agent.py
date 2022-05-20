import numpy as np
from data.graph import AdjList_Chicago
from collections import defaultdict
from simulator.config import *
import torch as T

# Device configuration
device = T.device('cuda' if T.cuda.is_available() else 'cpu')
print('The device is: ', device)

N_ACTIONS = 10

class Agent(object):
    def __init__(self):
        self._reward_maker = None

    def set_reward_scheme(self, scheme):
        self._reward_maker = scheme

    def iterate_zones_reward(self, drivers):
        assert self._reward_maker is not None
        rewards = defaultdict(lambda: defaultdict(lambda:0))
        for reward_zid in range(1, TOTAL_ZONES+1):
            for act in range(N_ACTIONS):
                rewards[reward_zid][act] = self._reward_maker.reward_scheme(drivers, reward_zid, act)
        return rewards

    @staticmethod
    def _one_hot_encode(x, length):
        code = [0] * length
        code[x] = 1
        return code

    # typically used by zone based method
    @classmethod
    def get_state_local_dist(cls, zone_id, demand_dist, supply_dist):
        max_adj_num = 10  # include itself
        adj_num = len(AdjList_Chicago[zone_id])  # not include itself
        adj_demand_supply_diff = np.zeros(max_adj_num)
        adj_demand_supply_diff[0] = demand_dist[zone_id] - supply_dist[zone_id]
        for i in range(adj_num):
            adj_zone = AdjList_Chicago[zone_id][i]
            adj_demand_supply_diff[i + 1] = demand_dist[adj_zone] - supply_dist[adj_zone]
        # print(adj_demand_supply_diff)
        adj_diff_mean = np.mean(adj_demand_supply_diff[0:adj_num + 1])
        adj_diff_std = np.std(adj_demand_supply_diff[0:adj_num + 1])
        if adj_diff_std == 0:
            adj_demand_supply_diff = np.zeros(max_adj_num)
        else:
            for i in range(adj_num + 1):
                adj_demand_supply_diff[i] = (adj_demand_supply_diff[i] - adj_diff_mean) / adj_diff_std
        zone_code = np.array(cls._one_hot_encode(zone_id - 1, 77))  # id 1, endcode as 0 here [1,0,0,0,....]
        return np.concatenate([zone_code, adj_demand_supply_diff])

if __name__ == '__main__':
    demands = np.random.randint(1,10,78)
    supplies = np.random.randint(1,10,78)
    print(demands)
    print(supplies)

    state = Agent.get_state_local_dist(2, demands, supplies)
    print(state)
    print(len(state))