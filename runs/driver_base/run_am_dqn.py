from simulator.env import Env
from simulator.timer import Timer
from simulator.objects import Reward_Distribution_v2
from simulator.config import *
from simulator.monitor import Monitor
from algorithms.driver_base.am_dqn import AM_DQN_Agent

RUN_STEP = 1027180

def run_am_dqn():
    env = Env()
    env.set_reward_scheme(Reward_Distribution_v2())
    agent = AM_DQN_Agent(1678, 10, 256, 0.0001)
    #agent = AM_DQN_Agent(1524, 10, 256, 0.0001)
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
            actions = agent.select_action(obs, env.monitor_drivers, i_step)
            next_obs, rewards, done, _ = env.step(actions)
            if ON_MONITOR:
                Monitor.reset_by_zone()
            agent.store_exp(env.monitor_drivers, obs, actions, rewards, next_obs)
            agent.update(i_step)
            obs = next_obs
            i_step += 1
        print("save checkpoint")
        agent.policy_net.save_checkpoint()
        print("Episode end:")
        print("The current time stamp: ", Timer.get_time_step())
        print("The current step: ", i_step)
        print(env.show_metrics_in_summary())


if __name__ == '__main__':
    run_am_dqn()