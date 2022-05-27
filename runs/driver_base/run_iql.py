from simulator.env import Env
from algorithms.driver_base.rewards import Reward_ICAART, Reward_SD_DIST
from algorithms.driver_base.iql import IQL_Agent

RUN_STEP = 3027180

def run_iql():
    env = Env()
    agent = IQL_Agent()
    agent.set_reward_scheme(Reward_SD_DIST())
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
            locs = obs["driver_locs"]
            actions = agent.select_action_softmax(env.monitor_drivers, i_step)
            obs, _, done, _ = env.step(actions)
            rewards = agent.iterate_drivers_reward(env.monitor_drivers, actions, obs)
            agent.update(env.monitor_drivers, actions, rewards, locs)
            i_step += 1
        #print("save Json")
        #agent.save()
        print("Episode end:")
        print("The current step: ", i_step)
        print(env.show_metrics_in_summary())

if __name__ == "__main__":
    run_iql()