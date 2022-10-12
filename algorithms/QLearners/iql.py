from simulator.timer import Timer
from simulator.config import *
from collections import defaultdict
from algorithms.agent import Agent
import torch.nn.functional as F
from torch.distributions import Categorical
import torch as T
import numpy as np
import random
import json

EPSILON= 0.1
GAMMA = 0.99
ALPHA = 0.1
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 20
N_ACTIONS = 10

LOAD = False

class IQL_Agent(Agent):
    def __init__(self):
        super(IQL_Agent, self).__init__()
        self.Q = defaultdict(lambda : np.zeros(N_ACTIONS))
        self.delta = defaultdict(lambda : 0)
        if LOAD:
            self.load()

    def _get_next_state(self, driver):
        if driver.in_service:
            return None
        else:
            return (Timer.get_time(Timer.get_time_step()), Timer.get_day(Timer.get_time_step()), driver.zid)

    def select_action_argmax(self, drivers, steps_done):
        actions = [-1] * len(drivers)
        time = Timer.get_time(Timer.get_time_step())
        day = Timer.get_day(Timer.get_time_step())
        assert 0<=time<=1440
        assert 1<=day<=7

        eps_threshold = 0.1 #EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        for did, driver in drivers.items():
            if driver.on_line is True:
                if random.random() > eps_threshold:
                    actions[did] = np.argmax(self.Q[(time, day, driver.zid)])
                else:
                    actions[did] = random.randrange(N_ACTIONS)
        return actions

    def select_action_softmax(self, drivers, steps_done):
        actions = [-1] * len(drivers)
        time = Timer.get_time(Timer.get_time_step())
        day = Timer.get_day(Timer.get_time_step())
        assert 0<=time<=1440
        assert 1<=day<=7

        for did, driver in drivers.items():
            if driver.on_line is True:
                probs = F.softmax(T.tensor(self.Q[(time, day, driver.zid)]), dim=0)
                m = Categorical(probs)
                action = m.sample()
                actions[did] = action.item()
        return actions

    def select_action_softmax_mask(self, drivers, steps_done):
        actions = [-1] * len(drivers)
        time = Timer.get_time(Timer.get_time_step())
        day = Timer.get_day(Timer.get_time_step())
        assert 0<=time<=1440
        assert 1<=day<=7

        for did, driver in drivers.items():
            if driver.on_line is True:
                adj_num = self.get_adj_zone_num(driver.zid)
                probs = F.softmax(T.tensor(self.Q[(time, day, driver.zid)][0:adj_num+1]), dim=0)
                m = Categorical(probs)
                action = m.sample()
                actions[did] = action.item()
        return actions


    def update(self, drivers, actions, rewards, locs):
        assert len(actions) == len(rewards)
        time = Timer.get_time(Timer.get_time_step()-1)
        day = Timer.get_day(Timer.get_time_step()-1)
        assert 0 <= time <= 1440
        assert 1 <= day <= 7
        for did, driver in drivers.items():
            A = actions[did]
            if A != -1:
                assert rewards[did] is not None
                assert 1<=locs[did]<=77
                S = (time, day, locs[did])
                S_pi = self._get_next_state(driver)
                R = rewards[did]
                if S_pi is not None:
                    self.delta[S] = abs(R + GAMMA * np.max(self.Q[S_pi]) - self.Q[S][A])
                    self.Q[S][A] = self.Q[S][A] + ALPHA * (R + GAMMA * np.max(self.Q[S_pi]) - self.Q[S][A])
                else:
                    self.delta[S] = abs(R - self.Q[S][A])
                    self.Q[S][A] = self.Q[S][A] + ALPHA * (R - self.Q[S][A])

    def save(self):
        print('Saving......')
        data = [{'key': k, 'value': v.tolist()} for k, v in self.Q.items()]
        #print('save:', data)
        with open('../checkpoints/qleaning.json', 'w') as fp:
            json.dump(data, fp)
        print('Save Done!')

    def load(self):
        print('Loading......')
        with open('../checkpoints/qleaning.json', 'r') as fp:
            data=json.load(fp)
        for elem in data:
            S=tuple(elem['key'])
            self.Q[S]=np.array(elem['value'])
        #print('load:',self.Q)
        print('Load Done!')

    def get_value_function(self):
        Vs = []
        time = Timer.get_time(Timer.get_time_step()-1)
        day = Timer.get_day(Timer.get_time_step()-1)

        for zid in range(1, TOTAL_ZONES+1):
            q_values = np.array(self.Q[(time, day, zid)])
            probs = F.softmax(T.tensor(q_values), dim=0).numpy()
            Vs.append(np.sum(q_values*probs))
        return Vs

    def get_delta_value(self):
        delta = []
        time = Timer.get_time(Timer.get_time_step() - 1)
        day = Timer.get_day(Timer.get_time_step() - 1)

        for zid in range(1, TOTAL_ZONES + 1):
           delta.append(self.delta[(time, day, zid)])
        return delta


if __name__ == "__main__":
    agent=IQL_Agent()
    agent.load()
    #print(agent.Q)

    print(agent.get_value_function(5,4,7))




