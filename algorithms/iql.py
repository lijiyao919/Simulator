from simulator.timer import Timer
from collections import defaultdict
import numpy as np
import random

EPSILON= 0.1
GAMMA = 0.99
ALPHA = 0.1

class IQL_Agent:
    def __init__(self):
        self.Q = defaultdict(lambda : np.zeros(10))
        self._directions = [i for i in range(10)]

    def _get_state(self, driver):
        return (Timer.get_time(Timer.get_time_step()), Timer.get_day(Timer.get_time_step()), driver.zid)

    def _get_next_state(self, driver):
        if driver.in_service:
            return (Timer.get_time(driver.wake_up_time), Timer.get_day(driver.wake_up_time), driver.zid)
        else:
            return (Timer.get_time(Timer.get_time_step()+1), Timer.get_day(Timer.get_time_step()+1), driver.zid)

    def select_action(self, drivers):
        actions = [-1] * len(drivers)
        for did, driver in drivers.items():
            if driver.on_line is True:
                if random.uniform(0, 1) < EPSILON:
                    actions[did] = random.choice(self._directions)
                else:
                    actions[did] = np.argmax(self.Q[self._get_state(driver)])
        return actions


    def update(self, drivers, actions, rewards):
        assert len(actions) == len(rewards)
        for did, driver in drivers.items():
            A = actions[did]
            if A != -1:
                assert rewards[did] is not None
                S = self._get_state(driver)
                S_pi = self._get_next_state(driver)
                R = rewards[did]
                self.Q[S][A] = self.Q[S][A] + ALPHA * (R + GAMMA * np.max(self.Q[S_pi]) - self.Q[S][A])


if __name__ == "__main__":
    agent=IQL_Agent()
    print(agent._q_table[0])