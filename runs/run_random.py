from simulator.env import Env
import random

def run_random():
    env = Env()
    env.reset()
    #print(env.show_graph())
    #print(env.show_drivers_in_spatial())
    directions = [i for i in range(10)]
    for _ in range(44641):
        actions = [random.choice(directions) for _ in range(env.get_drivers_length())]
        env.step(actions)
        #print(actions[7])
        #print(env.monitor_drivers[7])
        #print(env.show_riders_in_spatial())
        #print(env.show_drivers_in_spatial())
    print(env.show_metrics_in_spatial())
    print(env.show_metrics_each_driver())
    print(env.show_metrics_in_summary())

if __name__ == "__main__":
    run_random()