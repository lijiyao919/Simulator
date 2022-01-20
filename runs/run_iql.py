from simulator.env import Env
from simulator.timer import Timer
from simulator.config import *
from simulator.objects import Reward_ICAART
from algorithms.iql import IQL_Agent

RUN_STEP = 5000000

def run_iql():
    env = Env()
    env.set_reward_scheme(Reward_ICAART())
    agent = IQL_Agent(308)
    i_step = 0

    while i_step < RUN_STEP:
        env.reset()
        done = False
        while not done:
            if Timer.get_time_step() % TOTAL_MINUTES_ONE_DAY == 0:
                print("The current time stamp: ", Timer.get_time_step())
                print("The current date: ", Timer.get_date(Timer.get_time_step()))
            locs = agent.record_current_loc(env.monitor_drivers)
            actions = agent.select_action(env.monitor_drivers, i_step)
            _, rewards, done, _ = env.step(actions)
            agent.update(env.monitor_drivers, actions, rewards, locs)
            if i_step % TOTAL_MINUTES_ONE_DAY == 0:
                #print(agent.Q)
                #print(actions)
                #print(env.show_fail_riders_num_in_spatial())
                #print(env.show_drivers_num_in_spatial())
                print(env.show_metrics_in_summary())
            i_step += 1
        print("Episode end:\n")
        print(env.show_metrics_in_summary())

if __name__ == "__main__":
    run_iql()