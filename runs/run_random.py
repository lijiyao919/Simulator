from simulator.env import Env
import random

RUN_STEP = 100000

def run_random():
    env = Env()
    i_step = 0
    directions = [i for i in range(10)]

    while i_step < RUN_STEP:
        env.reset()
        done = False
        while not done:
            actions = [random.choice(directions) for _ in range(env.get_drivers_length())]
            _, _, done, _ = env.step(actions)
            if i_step % 1000 == 0:
                print(env.show_metrics_in_summary())
            i_step += 1
        print("Episode end:\n")
        print(env.show_metrics_in_summary())

if __name__ == "__main__":
    run_random()