from abc import ABC, abstractmethod
from data.graph import AdjList_Chicago

class Reward(ABC):
    @abstractmethod
    def reward_scheme(self, drivers, reward_zid, act):
        pass


class Zone_Reward(Reward):
    def reward_scheme(self, drivers, reward_zid, act):
        reward = 0
        for driver in drivers.values():
            if driver.reward_zid == reward_zid and driver.last_act==act:
                if driver.in_service:
                    reward += 1
                else:
                    reward -= 1
                driver.reward_zid = -1
                driver.last_act = -1
        return reward
