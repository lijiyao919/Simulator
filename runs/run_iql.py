from simulator.env import Env
from simulator.objects import Reward_ICAART
from algorithms.iql import IQL_Agent

RUN_STEP = 100000

def run_iql():
    env = Env()
    env.set_reward_scheme(Reward_ICAART())
    agent = IQL_Agent()
    i_step = 0

    while i_step < RUN_STEP:
        env.reset()
        done = False
        while not done:
            actions = agent.select_action(env.monitor_drivers)
            _, rewards, done, _ = env.step(actions)
            agent.update(env.monitor_drivers, actions, rewards)
            i_step += 1
            if i_step % 1000 == 0:
                print(env.show_metrics_in_summary())

if __name__ == "__main__":
    run_iql()