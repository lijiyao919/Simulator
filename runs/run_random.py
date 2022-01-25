from simulator.env import Env
from simulator.timer import Timer
from simulator.config import *
import random

RUN_STEP = 100000
N_ACTIONS = 10

def run_random():
    env = Env()
    i_step = 0

    while i_step < RUN_STEP:
        env.reset()
        done = False
        while not done:
            if Timer.get_time_step() != 0 and Timer.get_time_step() % TOTAL_MINUTES_ONE_DAY == 0:
                print("The current step: ", i_step)
                print("The current time stamp: ", Timer.get_time_step())
                print("The current date: ", Timer.get_date(Timer.get_time_step()))
                print(env.show_metrics_in_summary())
            actions = [random.randrange(N_ACTIONS) for _ in range(env.get_drivers_length())]
            _, _, done, _ = env.step(actions)
            i_step += 1
        print("Episode end:")
        print(env.show_metrics_in_summary())

if __name__ == "__main__":
    run_random()