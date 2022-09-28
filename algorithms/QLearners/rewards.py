from abc import ABC, abstractmethod
from simulator.config import *

class Reward(ABC):
    @abstractmethod
    def reward_scheme(self, driver, obs):
        pass

#from ICAART2022 paper
class Reward_ICAART(Reward):
    def reward_scheme(self, driver, obs):
        if driver.in_service is True:
            return 2
        elif driver.on_line is True:
            return -1
        elif driver.on_line is False:
            return -2
        else:
            raise Exception("wrong reward")

#from ITSC2022 paper
class Reward_Distribution_v2(Reward):
    def reward_scheme(self, driver, obs):
        if driver.in_service is True:
            assert driver.rider is not None
            return (PATIENCE_TIME - driver.rider.call_taxi_duration)
        else:
            return 0

class Reward_SD_DIST(Reward):
    def reward_scheme(self, driver, obs):
        return obs["on_call_rider_num"][driver.zid] - obs["online_driver_num"][driver.zid]
