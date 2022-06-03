import numpy as np

from simulator.env import Env
from data.graph import AdjList_Chicago
from simulator.monitor import Monitor
from simulator.config import *
from torch.distributions import Categorical
import torch as T
import torch.nn.functional as F
from simulator.timer import Timer

POLICY = "greedy"

def select_action(obs, drivers):
    actions = [-1] * len(drivers)

    for did, driver in drivers.items():
        if driver.on_line is True:
            assert obs["driver_locs"][did] == driver.zid
            choice_num = len(AdjList_Chicago[driver.zid]) +1
            choices = [0]*choice_num
            for i in range(choice_num):
                if i==choice_num-1:
                    choices[i] = obs["on_call_rider_num"][driver.zid] - obs["online_driver_num"][driver.zid]
                else:
                    adj_zone = AdjList_Chicago[driver.zid][i]
                    choices[i] = obs["on_call_rider_num"][adj_zone] - obs["online_driver_num"][adj_zone]
            if POLICY=="softmax":
                probs = F.softmax(T.tensor([choices], dtype=float), dim=1)
                m = Categorical(probs)
                actions[did] = m.sample()
            elif POLICY=="greedy":
                actions[did] = np.argmax(choices)
            else:
                raise Exception("No such policy.")

    return actions

def run_demand_base():
    env = Env()
    i_step = 0

    obs = env.reset()
    done = False
    while not done:
        '''if Timer.get_time_step() != 0 and Timer.get_time_step() % TOTAL_MINUTES_ONE_DAY == 0:
            print("The current step: ", i_step)
            print("The current time stamp: ", Timer.get_time_step())
            print("The current date: ", Timer.get_date(Timer.get_time_step()))
            print(env.show_metrics_in_summary())'''
        actions = select_action(obs, env.monitor_drivers)
        next_obs, _, done, _ = env.step(actions)
        obs = next_obs
        i_step += 1
    print("Episode end:")
    print("The current step: ", i_step)
    print("The current time stamp: ", Timer.get_time_step())
    print(env.show_metrics_in_summary())

if __name__ == "__main__":
    run_demand_base()