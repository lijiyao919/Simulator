import matplotlib.pyplot as plt
import numpy as np
import torch as T
from simulator.timer import Timer

class Monitor:
    ep_cnt = 0

    @staticmethod
    def init():
        Monitor.ep_cnt += 1
        plt.ion()

    @staticmethod
    def plot_rider_driver_num_by_time(calls, drivers, matches):
        plt.figure(1)
        plt.clf()
        calls_t = T.tensor(calls, dtype=T.int)
        drivers_t = T.tensor(drivers, dtype=T.int)
        matches_t = T.tensor(matches, dtype=T.int)
        plt.title('Episode: '+str(Monitor.ep_cnt))
        plt.xlabel('Time')
        plt.ylabel('Number#')
        plt.plot(calls_t.numpy(), label="CALL_NUM")
        plt.plot(drivers_t.numpy(), label="DRIVER_NUM")
        plt.plot(matches_t.numpy(), label="MATCH_NUM")
        plt.legend()
        plt.pause(0.001)  # pause a bit so that plots are updated

    @staticmethod
    def plot_rider_driver_num_by_zone(calls, drivers):
        plt.figure(2)
        plt.clf()
        zones = np.arange(1,78)
        plt.title('Time: ' + str(Timer.get_time(Timer.get_time_step())))
        plt.xticks([i for i in range(1, 78, 2)])
        plt.xlabel('Time')
        plt.ylabel('Number#')
        plt.bar(zones-0.1, calls, width=0.3, label="CALL_NUM")
        plt.bar(zones+0.2, drivers, width=0.3, label="DRIVER_NUM")
        plt.legend()
        plt.pause(0.001)  # pause a bit so that plots are updated

    @staticmethod
    def close():
        plt.ioff()
        plt.show()

