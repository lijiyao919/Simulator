import random
from simulator.config import *
import numpy as np

from simulator.env import Env
from data.graph import AdjList_Chicago
from simulator.timer import Timer

epsilon = 0

def select_action(obs, drivers):
    actions = [-1] * len(drivers)

    for did, driver in drivers.items():
        if driver.on_line is True:
            assert obs["driver_locs"][did] == driver.zid
            choice_num = len(AdjList_Chicago[driver.zid]) +1
            choices = [0]*choice_num
            for i in range(choice_num):
                if i==choice_num-1:
                    choices[i] = obs["on_call_rider_num"][driver.zid]
                else:
                    adj_zone = AdjList_Chicago[driver.zid][i]
                    choices[i] = obs["on_call_rider_num"][adj_zone]
            if random.random() >= epsilon:
                actions[did] = np.argmax(choices)
            else:
                actions[did] = random.randrange(N_ACTIONS)

    return actions

def run_greedy():
    env = Env()
    i_step = 0

    env.reset()
    done = False
    while not done:
        obs = env.pre_step()
        actions = select_action(obs, env.monitor_drivers)
        next_obs, _, done, _ = env.step(actions)
        i_step += 1
    print("Episode end:")
    print("The current step: ", i_step)
    print("The current time stamp: ", Timer.get_time_step())
    print(env.show_metrics_in_summary())

if __name__ == "__main__":
    run_greedy()