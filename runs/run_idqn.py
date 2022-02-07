from simulator.env import Env
from simulator.timer import Timer
from simulator.objects import Reward_ICAART
from simulator.config import *
from algorithms.idqn import IDQN_Agent

RUN_STEP = 178641

def run_idqn():
    env = Env()
    env.set_reward_scheme(Reward_ICAART())
    agent = IDQN_Agent(1524, 10, 256, 0.0001, batch_size=32, target_update_feq=1000, eps_end=0.1, eps_decay=20, buffer_size=10000)
    agent.train_mode()
    i_step = 0

    while i_step < RUN_STEP:
        obs = env.reset()
        done = False
        while not done:
            if Timer.get_time_step() != 0 and Timer.get_time_step() % TOTAL_MINUTES_ONE_DAY == 0:
                print("The current step: ", i_step)
                print("The current time stamp: ", Timer.get_time_step())
                print("The current date: ", Timer.get_date(Timer.get_time_step()))
                print(env.show_metrics_in_summary())
            actions = agent.select_action(obs, env.monitor_drivers, i_step)
            next_obs, rewards, done, _ = env.step(actions)
            agent.store_exp(env.monitor_drivers, obs, actions, rewards, next_obs)
            agent.update(i_step)
            obs = next_obs
            i_step += 1
        print("Episode end:")
        print(env.show_metrics_in_summary())


if __name__ == '__main__':
    run_idqn()