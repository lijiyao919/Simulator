from abc import ABC, abstractmethod
from simulator.config import *

class Reward(ABC):
    @abstractmethod
    def reward_scheme(self, driver):
        pass

#from ICAART2022 paper
class Reward_ICAART(Reward):
    def reward_scheme(self, driver):
        if driver.in_service is True:
            return 2
        elif driver.on_line is True:
            return -1
        elif driver.on_line is False:
            return -2
        else:
            raise Exception("wrong reward")

#from ITSC2022 paper
class Reward_Distribution(Reward):
    def reward_scheme(self, driver):
        if driver.in_service is True:
            assert driver.rider is not None
            return (PATIENCE_TIME - driver.rider.call_taxi_duration)
        elif driver.on_line is True:
            return -1
        elif driver.on_line is False:
            return -2
        else:
            raise Exception("wrong reward")

class Reward_Distribution_v2(Reward):
    def reward_scheme(self, driver):
        if driver.in_service is True:
            assert driver.rider is not None
            return (PATIENCE_TIME - driver.rider.call_taxi_duration)
        else:
            return 0
