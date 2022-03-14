from simulator.env import Env
from simulator.timer import Timer
from simulator.config import *
from simulator.objects import Reward_ICAART
from algorithms.iql import IQL_Agent

RUN_STEP = 1027180

def run_iql():
    env = Env()
    env.set_reward_scheme(Reward_ICAART())
    agent = IQL_Agent()
    i_step = 0

    while i_step < RUN_STEP:
        obs = env.reset()
        done = False
        while not done:
            '''if Timer.get_time_step() != 0 and Timer.get_time_step() % TOTAL_MINUTES_ONE_DAY == 0:
                print("The current step: ", i_step)
                print("The current time stamp: ", Timer.get_time_step())
                print("The current date: ", Timer.get_date(Timer.get_time_step()))
                print(env.show_metrics_in_summary())'''
            locs = obs["driver_locs"]
            actions = agent.select_action(env.monitor_drivers, i_step)
            obs, rewards, done, _ = env.step(actions)
            agent.update(env.monitor_drivers, actions, rewards, locs)
            i_step += 1
        print("save Json")
        agent.save()
        print("Episode end:")
        print("The current step: ", i_step)
        print(env.show_metrics_in_summary())

if __name__ == "__main__":
    run_iql()