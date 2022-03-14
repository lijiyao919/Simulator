from simulator.timer import Timer
from collections import defaultdict
import numpy as np
import random
import json
import math

EPSILON= 0.1
GAMMA = 0.99
ALPHA = 0.1
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 20
N_ACTIONS = 10

LOAD = True

class IQL_Agent:
    def __init__(self):
        self.Q = defaultdict(lambda : np.zeros(N_ACTIONS))
        if LOAD:
            self.load()

    def _get_next_state(self, driver):
        if driver.in_service:
            return (Timer.get_time(driver.wake_up_time), Timer.get_day(driver.wake_up_time), driver.zid)
        else:
            return (Timer.get_time(Timer.get_time_step()), Timer.get_day(Timer.get_time_step()), driver.zid)

    def select_action(self, drivers, steps_done):
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
                #print(did, S, S_pi, R)
                self.Q[S][A] = self.Q[S][A] + ALPHA * (R + GAMMA * np.max(self.Q[S_pi]) - self.Q[S][A])

    def save(self):
        print('Saving......')
        data = [{'key': k, 'value': v.tolist()} for k, v in self.Q.items()]
        #print('save:', data)
        with open('checkpoints/qleaning.json', 'w') as fp:
            json.dump(data, fp)
        print('Save Done!')

    def load(self):
        print('Loading......')
        with open('checkpoints/qleaning.json', 'r') as fp:
            data=json.load(fp)
        for elem in data:
            S=tuple(elem['key'])
            self.Q[S]=np.array(elem['value'])
        #print('load:',self.Q)
        print('Load Done!')


if __name__ == "__main__":
    agent=IQL_Agent()
    print(agent.Q)
