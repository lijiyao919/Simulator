import numpy as np
from simulator.timer import Timer
from collections import defaultdict
from data.graph import AdjList_Chicago
import torch as T

# Device configuration
device = T.device('cuda' if T.cuda.is_available() else 'cpu')
print('The device is: ', device)

class Agent(object):
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
            #S_t+n transition
            #return cls.get_state(Timer.get_time(driver.wake_up_time), Timer.get_day(driver.wake_up_time), driver.zid)
            #final transition
            return None
        else:
            return cls.get_state(Timer.get_time(Timer.get_time_step()), Timer.get_day(Timer.get_time_step()), driver.zid)

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

