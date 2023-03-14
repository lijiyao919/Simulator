from simulator.env import Env
from simulator.timer import Timer
from simulator.config import *
import random


def run_random():
    env = Env()
    i_step = 0

    env.reset()
    done = False
    while not done:
        env.pre_step()
        actions = [random.randrange(N_ACTIONS) for _ in range(env.get_drivers_length())]
        _, _, done, _ = env.step(actions)
        i_step += 1
    print("Episode end:")
    print("The current step: ", i_step)
    print(env.show_metrics_in_summary())

if __name__ == "__main__":
    run_random()