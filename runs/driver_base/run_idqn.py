from simulator.env import Env
from algorithms.driver_base.rewards import Reward_ICAART, Reward_Distribution
from algorithms.driver_base.idqn import IDQN_Agent
from simulator.timer import Timer
from simulator.config import *

RUN_STEP = 1027180

def run_idqn():
    env = Env()
    agent = IDQN_Agent(1524, 10, 256, 0.0001, batch_size=32, target_update_feq=10000, eps_end=0.1, eps_decay=20, buffer_size=1000)
    agent.set_reward_scheme(Reward_ICAART())
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
                print(env.show_metrics_in_summary())'''
            actions = agent.select_action(obs, env.monitor_drivers, i_step)
            next_obs, _, done, _ = env.step(actions)
            rewards = agent.iterate_drivers_reward(env.monitor_drivers, actions, next_obs)
            agent.store_exp(env.monitor_drivers, obs, actions, rewards, next_obs)
            agent.update(i_step)
            obs = next_obs
            i_step += 1
        print("save checkpoint")
        agent.policy_net.save_checkpoint()
        print("Episode end:")
        print("The current step: ", i_step)
        print(env.show_metrics_in_summary())


if __name__ == '__main__':
    run_idqn()