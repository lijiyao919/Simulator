from abc import ABC, abstractmethod
from simulator.config import *
import math

class Reward(ABC):
    @abstractmethod
    def reward_scheme(self, driver, info):
        pass

#from ICAART2022 paper
class Reward_ICAART(Reward):
    def reward_scheme(self, driver, info):
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
    def reward_scheme(self, driver, info):
        if driver.in_service is True:
            assert driver.rider is not None
            return (PATIENCE_TIME - driver.rider.call_taxi_duration)
        else:
            return 0

class Reward_COOP(Reward):
    def reward_scheme(self, driver, info):
        #dist_diff = obs["on_call_rider_num"][driver.zid] - obs["online_driver_num"][driver.zid]
        if driver.in_service is True:
            assert driver.rider is not None
            coop_reg = info["fail_math_rate"][driver.pickup_zid]
            assert 0 <= coop_reg <= 1
            return 1-coop_reg #2-coop_reg
        else:
            return -1 #0.63*math.atan(0.05*dist_diff)
