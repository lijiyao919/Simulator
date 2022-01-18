from simulator.env import Env
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
            actions = [random.randrange(N_ACTIONS) for _ in range(env.get_drivers_length())]
            _, _, done, _ = env.step(actions)
            if i_step % 1440 == 0:
                #print(actions)
                #print(env.show_fail_riders_num_in_spatial())
                #print(env.show_offline_driver_status_in_specific_zone(9))
                print(env.show_metrics_in_summary())
            i_step += 1
        print("Episode end:\n")
        print(env.show_metrics_in_summary())

if __name__ == "__main__":
    run_random()