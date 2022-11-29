from simulator.env import Env
from algorithms.rewards import Reward_COOP
from algorithms.PGs.a2c import A2C_Agent
import timeit

RUN_STEP = 20000


def run_a2c():
    env = Env()
    agent = A2C_Agent(327, 10, 256, 128, 0.001, batch_size=1024)
    agent.set_reward_scheme(Reward_COOP())
    agent.train_mode()
    i_step = 0

    while i_step < RUN_STEP:
        env.reset()
        done = False
        while not done:
            obs = env.pre_step()
            actions, log_probs, values, entropys = agent.feed_forward(obs, env.monitor_drivers)
            V = agent.read_V()
            next_obs, _, done, info = env.step(actions, V)
            rewards = agent.iterate_drivers_reward(env.monitor_drivers, actions, info)
            agent.store_exp(env.monitor_drivers, log_probs, values, rewards, next_obs, entropys, actions)
            agent.learn()
            i_step += 1
        #print("save checkpoint")
        #agent.policy_net.save_checkpoint()
        print("Episode end:")
        print("The current step: ", i_step)
        print(env.show_metrics_in_summary())

if __name__ == '__main__':
    run_a2c()