from simulator.env import Env
from simulator.timer import Timer
from algorithms.zone_base.rewards import Zone_Reward
from simulator.config import *
from simulator.monitor import Monitor
from algorithms.zone_base.dqn import DQN_Agent

RUN_STEP = 1027180

def run_dqn():
    env = Env()
    agent = DQN_Agent(87, 10, 128, 0.0001)
    agent.set_reward_scheme(Zone_Reward())
    agent.train_mode()
    i_step = 0

    while i_step < RUN_STEP:
        obs = env.reset()
        done = False
        while not done:
            '''if Timer.get_time_step() != 0 and Timer.get_time_step() % TOTAL_MINUTES_ONE_DAY == 0:
                print("The current step: ", i_step)
                print("The current time stamp: ", Timer.get_time_step())
                print("The current date: ", Timer.get_date(Timer.get_time_step()))
                print(env.show_metrics_in_summary())
                if ON_MONITOR:
                    Monitor.reset_by_time()'''
            driver_actions = agent.select_actions(env.monitor_drivers, obs)
            next_obs, rewards, done, _ = env.step(driver_actions)
            if ON_MONITOR:
                Monitor.reset_by_zone()
            rewards = agent.iterate_zones_reward(env.monitor_drivers)
            agent.store_exp(obs, rewards, next_obs)
            agent.update(i_step)
            obs = next_obs
            i_step += 1
        #print("save checkpoint")
        #agent.policy_net.save_checkpoint()
        print("Episode end:")
        print("The current time stamp: ", Timer.get_time_step())
        print("The current step: ", i_step)
        print(env.show_metrics_in_summary())


if __name__ == '__main__':
    run_dqn()