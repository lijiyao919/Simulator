from simulator.env import Env
from algorithms.rewards import Reward_COOP
from algorithms.PGs.a2c import A2C_Agent

RUN_STEP = 3027180


def run_a2c():
    env = Env()
    agent = A2C_Agent(1524, 10, 256, 128, 0.00001)
    agent.set_reward_scheme(Reward_COOP())
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
            actions, log_probs, values, entropys = agent.feed_forward(obs, env.monitor_drivers)
            next_obs, _, done, info = env.step(actions)
            rewards = agent.iterate_drivers_reward(env.monitor_drivers, actions, info)
            agent.store_exp(env.monitor_drivers, log_probs, values, rewards, next_obs, entropys, actions)
            agent.learn()
            obs = next_obs
            i_step += 1
        #print("save checkpoint")
        #agent.policy_net.save_checkpoint()
        print("Episode end:")
        print("The current step: ", i_step)
        print(env.show_metrics_in_summary())

if __name__ == '__main__':
    run_a2c()