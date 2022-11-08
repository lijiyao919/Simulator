from simulator.env import Env
from algorithms.rewards import Reward_COOP
from algorithms.PGs.ppo import PPO_Agent
import timeit

RUN_STEP = 20000


def run_ppo():
    env = Env()
    agent = PPO_Agent(327, 10, 256, 128, 0.00001, batch_size=1024, ppo_step=4, clip_param=0.1)
    agent.set_reward_scheme(Reward_COOP())
    agent.train_mode()
    i_step = 0

    while i_step < RUN_STEP:
        obs = env.reset()
        done = False
        while not done:
            actions, log_probs, values = agent.feed_forward(obs, env.monitor_drivers)
            next_obs, _, done, info = env.step(actions)
            rewards = agent.iterate_drivers_reward(env.monitor_drivers, actions, info)
            agent.store_exp(env.monitor_drivers, actions, log_probs, values, rewards, next_obs)
            agent.learn()
            obs = next_obs
            i_step += 1
        #print("save checkpoint")
        #agent.policy_net.save_checkpoint()
        print("Episode end:")
        print("The current step: ", i_step)
        print(env.show_metrics_in_summary())

if __name__ == '__main__':
    run_ppo()