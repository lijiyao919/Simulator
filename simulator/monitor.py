import matplotlib.pyplot as plt
import torch as T

class Monitor:
    ep_cnt = 0

    @staticmethod
    def init():
        Monitor.ep_cnt += 1
        plt.ion()

    @staticmethod
    def plot_success_match(calls, drivers, matches):
        plt.figure(2)
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
    def close():
        plt.ioff()
        plt.show()

