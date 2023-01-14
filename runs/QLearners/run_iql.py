from simulator.env import Env
from algorithms.rewards import Reward_ICAART
from algorithms.QLearners.iql import IQL_Agent
from simulator.monitor import Monitor
from simulator.config import *

ACT_SELECT="softmax"

RUN_STEP = 10000

def run_iql():
    env = Env()
    agent = IQL_Agent()
    agent.set_reward_scheme(Reward_ICAART())
    i_step = 0

    while i_step < RUN_STEP:
        env.reset()
        done = False
        while not done:
            obs = env.pre_step()
            locs = obs["driver_locs"]
            if ACT_SELECT=="argmin":
                actions = agent.select_action_argmax(env.monitor_drivers, i_step)
            elif ACT_SELECT=="softmax":
                actions = agent.select_action_softmax(env.monitor_drivers, i_step)
            elif ACT_SELECT=="softmax_mask":
                actions = agent.select_action_softmax_mask(env.monitor_drivers, i_step)
            else:
                raise Exception("No such action.")
            next_obs, _, done, _ = env.step(actions)
            rewards = agent.iterate_drivers_reward(env.monitor_drivers, actions)
            agent.update(env.monitor_drivers, actions, rewards, locs)
            i_step += 1
        #print("save Json")
        #agent.save()
        print("Episode end:")
        print("The current step: ", i_step)
        print(env.show_metrics_in_summary())

if __name__ == "__main__":
    run_iql()