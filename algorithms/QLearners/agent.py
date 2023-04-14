import numpy as np
from simulator.timer import Timer
from collections import defaultdict
from data.graph import AdjList_Chicago
import torch as T

# Device configuration
device = T.device('cuda' if T.cuda.is_available() else 'cpu')
print('The device is: ', device)

class Agent(object):
    def __init__(self):
        self._reward_maker = None

    def set_reward_scheme(self, scheme):
        self._reward_maker = scheme

    def iterate_drivers_reward(self, drivers, actions, info=None):
        assert self._reward_maker is not None
        rewards = [None]*len(drivers)
        for did, driver in drivers.items():
            if actions[did] != -1: # make sure driver take action
                rewards[did] = self._reward_maker.reward_scheme(driver, info)
        return rewards

    def get_adj_zone_num(self, zid):
        return len(AdjList_Chicago[zid])

    def softmax(self, x):
        return (np.exp(x) / np.exp(x).sum())

    @staticmethod
    def _binary_encode(x, length):
        return [int(d) for d in str(bin(x))[2:].zfill(length)]

    @staticmethod
    def _one_hot_encode(x, length):
        code = [0] * length
        code[x] = 1
        return code

    @classmethod
    def get_state(cls, time, day, zone_id):
        # time_bin = np.array(cls._binary_encode(time, 11))
        time_code = np.array(cls._one_hot_encode(time, 1440))
        # day_bin = np.array(cls._binary_encode(day, 3))
        day_code = np.array(cls._one_hot_encode(day - 1, 7))  # Mon(1), encode as 0 here, [1,0,0,0,...]
        # zid_bin = np.array(cls._binary_encode(zone_id, 7))
        zone_code = np.array(cls._one_hot_encode(zone_id - 1, 77))  # id 1, endcode as 0 here [1,0,0,0,....]
        '''print(time_code)
        print(day_code)
        print(zone_code)
        print(np.concatenate([time_code, day_code, zone_code]))'''
        return np.concatenate([time_code, day_code, zone_code])

    @classmethod
    def get_next_state(cls, driver):
        if driver.in_service:
            return None
        else:
            return cls.get_state(Timer.get_time(Timer.get_time_step()), Timer.get_day(Timer.get_time_step()), driver.zid)

    @classmethod
    def get_state_dist(cls, time, day, zone_id, demand_dist, supply_dist):
        time_code = np.array(cls._one_hot_encode(time, 1440))
        day_code = np.array(cls._one_hot_encode(day - 1, 7))  # Mon(1), encode as 0 here, [1,0,0,0,...]
        zone_code = np.array(cls._one_hot_encode(zone_id - 1, 77))  # id 1, endcode as 0 here [1,0,0,0,....]
        # dist_diff = np.array(demand_dist) - np.array(supply_dist)
        # dist_diff = np.delete(dist_diff / max(dist_diff.min(), dist_diff.max(), key=abs), [0])
        '''if max(np.array(demand_dist)) != 0:
            d_dist = np.delete(np.array(demand_dist) / max(np.array(demand_dist)), [0])
        else:
            d_dist = np.delete(np.array(demand_dist), [0])
        s_dist = np.delete(np.array(supply_dist) / max(np.array(supply_dist)), [0])'''
        d_dist = np.delete(np.array(demand_dist), 0)
        d_dist = (d_dist-np.mean(d_dist))/np.std(d_dist) if np.std(d_dist) != 0 else np.ones(77)

        s_dist = np.delete(np.array(supply_dist), 0)
        s_dist = (s_dist - np.mean(s_dist)) / np.std(s_dist) if np.std(s_dist) != 0 else np.ones(77)

        return np.concatenate([time_code, day_code, zone_code, d_dist, s_dist])

    @classmethod
    def get_next_state_dist(cls, driver, demand_dist, supply_dist):
        if driver.in_service:
            return None
        else:
            return cls.get_state_dist(Timer.get_time(Timer.get_time_step()), Timer.get_day(Timer.get_time_step()),
                                      driver.zid, demand_dist, supply_dist)

    @classmethod
    def get_state_dist_cmp(cls, time, day, zone_id, demand_dist, supply_dist):
        #time_code = np.array(cls._one_hot_encode(time, 1440))
        #day_code = np.array(cls._one_hot_encode(day - 1, 7))  # Mon(1), encode as 0 here, [1,0,0,0,...]
        zone_code = np.array(cls._one_hot_encode(zone_id - 1, 77))  # id 1, endcode as 0 here [1,0,0,0,....]

        cmp = []
        for i in range(1,78):
            if  0 <= (supply_dist[i] - demand_dist[i])/(1+supply_dist[i]) < 0.3:
                cmp += [1, 0, 0, 0, 0, 0]
            elif 0.3 <= (supply_dist[i] - demand_dist[i])/(1+supply_dist[i]) < 0.6:
                cmp += [0, 1, 0, 0, 0, 0]
            elif 0.6 <= (supply_dist[i] - demand_dist[i])/(1 + supply_dist[i]) <= 1:
                cmp += [0, 0, 1, 0, 0, 0]
            elif 0< (demand_dist[i]-supply_dist[i])/(1 + demand_dist[i]) <0.3:
                cmp += [0, 0, 0, 1, 0, 0]
            elif 0.3 <= (demand_dist[i]-supply_dist[i])/(1 + demand_dist[i]) <0.6:
                cmp += [0, 0, 0, 0, 1, 0]
            elif 0.6 <= (demand_dist[i]-supply_dist[i]) / (1 + demand_dist[i]) <=1:
                cmp += [0, 0, 0, 0, 0, 1]
            else:
                print(supply_dist[i], demand_dist[i])
                raise("Error in dist cmp")
        return np.concatenate([zone_code, cmp])

    @classmethod
    def get_next_state_dist_cmp(cls, driver, demand_dist, supply_dist):
        if driver.in_service:
            return None
        else:
            return cls.get_state_dist_cmp(Timer.get_time(Timer.get_time_step()), Timer.get_day(Timer.get_time_step()),
                                      driver.zid, demand_dist, supply_dist)


    def elimilate_actions_by_context(self, drivers, actions):
        trip_info = defaultdict(list)

        for did, driver in drivers.items():
            source = driver.zid
            # offline driver or stay still
            if actions[did] == -1 or actions[did]>=len(AdjList_Chicago[source]):
                continue
            destination = AdjList_Chicago[source][actions[did]]
            trip_key = str(source)+"->"+str(destination)
            trip_info[trip_key].append(did)

        #print(trip_info)
        for trip_key in trip_info.keys():
            source, destination = trip_key.split("->")
            col_trip_key = str(destination)+"->"+str(source)

            if col_trip_key not in trip_info.keys():
                continue

            #print(trip_key + ": " + str(trip_info[trip_key]))
            #print(col_trip_key + ": " + str(trip_info[col_trip_key]))

            if len(trip_info[trip_key]) == 0 or len(trip_info[col_trip_key]) == 0:
                continue
            if len(trip_info[trip_key]) == len(trip_info[col_trip_key]):
                #print("equal")
                for did in trip_info[trip_key]:
                    actions[did] = 9
                for did in trip_info[col_trip_key]:
                    actions[did] = 9
            elif len(trip_info[trip_key]) > len(trip_info[col_trip_key]):
                #print(">col")
                for did in trip_info[col_trip_key]:
                    actions[did] = 9
                for i in range(len(trip_info[col_trip_key])):
                    actions[trip_info[trip_key][i]] = 9
            elif len(trip_info[trip_key]) < len(trip_info[col_trip_key]):
                #print("<col")
                for did in trip_info[trip_key]:
                    actions[did] = 9
                for i in range(len(trip_info[trip_key])):
                    actions[trip_info[col_trip_key][i]] = 9
            else:
                raise Exception("action process mistakes.")
            trip_info[trip_key] = []
            trip_info[col_trip_key] = []

if __name__ == '__main__':
    demands = np.random.randint(1,10,78)
    supplies = np.random.randint(1,10,78)
    print(demands)
    print(supplies)