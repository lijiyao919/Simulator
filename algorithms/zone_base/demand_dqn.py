from algorithms.driver_base.idqn import IDQN_Agent
from algorithms.driver_base.agent import device
from simulator.timer import Timer
import torch.nn.functional as F
import torch as T
import numpy as np
from torch.distributions import Categorical

SAMPLE="softmax"

class DEMAND_DQN_Agent(IDQN_Agent):
    def __init__(self, input_dims, n_actions, fc1_dims, eta, buffer_size=10000, batch_size=32, gamma=0.99, target_update_feq=1000, eps_end=0.1, eps_decay=1000000):
        super().__init__(input_dims, n_actions, fc1_dims, eta, buffer_size, batch_size, gamma, target_update_feq, eps_end, eps_decay)


    def select_action(self, obs, drivers, steps_done):
        actions = [-1] * len(drivers)
        time = Timer.get_time(Timer.get_time_step())
        day = Timer.get_day(Timer.get_time_step())
        assert 0 <= time <= 1440
        assert 1 <= day <= 7

        for did, driver in drivers.items():
            if driver.on_line is True:
                assert obs["driver_locs"][did] == driver.zid
                with T.no_grad():
                    state = DEMAND_DQN_Agent.get_state_local_dist(driver.zid, obs["on_call_rider_num"], obs["online_driver_num"]) #dist
                    state_tensor = T.from_numpy(np.expand_dims(state.astype(np.float32), axis=0)).to(device)
                    if SAMPLE == "softmax":
                        #print("softmax")
                        probs = F.softmax(self.policy_net(state_tensor), dim=1)
                    elif SAMPLE =="softmax_rank":
                        #print("softmax_rank")
                        sort_idx = (-self.policy_net(state_tensor)).argsort(dim=1)
                        ranks = T.empty_like(sort_idx, dtype=T.float32)
                        ranks[0][sort_idx] = T.arange(1, len(sort_idx[0])+1, device=device, dtype=T.float32)
                        ranks = T.reciprocal(ranks)
                        probs = F.softmax(ranks, dim=1)
                    else:
                        raise Exception("No sampling type")
                    m = Categorical(probs)
                    action = m.sample()
                    actions[did] = action.item()

        return actions

