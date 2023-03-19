from simulator.env import Env
from algorithms.QLearners.rewards import Reward_ICAART, Reward_Distribution_v2, Reward_COOP
from algorithms.QLearners.iper import IPER_Agent
from simulator.timer import Timer
from simulator.config import *

RUN_STEP = 3027180

def run_iper():
    env = Env()
    agent = IPER_Agent(539, 10, 128, 0.0001, buffer_size=1000, batch_size=128, alpha=0.6, beta_start=0.4)
    agent.set_reward_scheme(Reward_COOP())
    agent.train_mode()
    i_step = 0

    while i_step < RUN_STEP:
        env.reset()
        done = False
        while not done:
            obs = env.pre_step()
            actions = agent.select_action(obs, env.monitor_drivers, i_step)
            next_obs, _, done, info = env.step(actions)
            rewards = agent.iterate_drivers_reward(env.monitor_drivers, actions, info)
            agent.store_exp(env.monitor_drivers, obs, actions, rewards, next_obs)
            agent.update(i_step)
            i_step += 1
        #print("save checkpoint")
        #agent.policy_net.save_checkpoint()
        print("Episode end:")
        print("The current step: ", i_step)
        print(env.show_metrics_in_summary())


if __name__ == '__main__':
    run_iper()