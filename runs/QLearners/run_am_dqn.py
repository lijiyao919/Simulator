from simulator.env import Env
from simulator.timer import Timer
from algorithms.QLearners.rewards import Reward_Distribution_v2
from simulator.config import *
from algorithms.QLearners.am_dqn import AM_DQN_Agent

RUN_STEP = 3027180

def run_am_dqn():
    env = Env()
    #agent = AM_DQN_Agent(308, 10, 128, 0.000001, target_update_feq=10000, buffer_size=100000, batch_size=32)
    agent = AM_DQN_Agent(539, 10, 256, 0.0001, target_update_feq=10000, buffer_size=100000, batch_size=32)
    agent.set_reward_scheme(Reward_Distribution_v2())
    agent.train_mode()
    i_step = 0

    while i_step < RUN_STEP:
        obs = env.reset()
        done = False
        while not done:
            actions = agent.select_action(obs, env.monitor_drivers, i_step)
            next_obs, _, done, _ = env.step(actions)
            rewards = agent.iterate_drivers_reward(env.monitor_drivers, actions)
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