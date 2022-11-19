import matplotlib.pyplot as plt
from simulator.config import *
import os
import datetime
import numpy as np
from simulator.timer import Timer

class Monitor:
    def __init__(self, graph):
        self._graph = graph
        self._supply = []
        self._demand = []
        self._optimal = []
        self._success = []

    def record_current_supply_demand(self):
        supply_now = 0
        demand_now = 0
        for zid in range(1, TOTAL_ZONES + 1):
            supply_now += len(self._graph[zid].drivers_on_line)
            demand_now += len(self._graph[zid].riders_on_call)
        self._supply.append(supply_now)
        self._demand.append(demand_now)

    def _compute_metrics_now(self):
        success_now = 0
        for zid in range(1, TOTAL_ZONES + 1):
            success_now += self._graph[zid].success_order_num_now
        self._success.append(success_now)

    def plot_metrics_by_time(self):
        self._compute_metrics_now()
        self._optimal.append(min(self._supply[-1], self._demand[-1]))
        plt.figure(1)
        plt.clf()
        plt.subplot(221)
        plt.plot(self._supply, label="SUPPLY#")
        plt.legend()

        plt.subplot(222)
        plt.plot(self._demand, label="DEMAND#")
        plt.legend()

        plt.subplot(223)
        plt.plot(self._optimal, label="OPTIMAL")
        plt.legend()

        plt.subplot(224)
        plt.plot(self._success, label="SUCCESS")
        plt.legend()
        plt.pause(0.001)  # pause a bit so that plots are updated
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

        '''folder_path = os.path.join(IMGS_FOLDER, 'time')
        os.makedirs(folder_path, exist_ok=True)
        path = os.path.join(folder_path, str(Timer.get_time_step()))
        plt.savefig(path)'''

    def reset_metrics(self):
        self._supply = []
        self._demand = []
        self._optimal = []
        self._success = []
        plt.ion()

    def clear_now_data(self):
        for zid in range(1, TOTAL_ZONES+1):
            self._graph[zid].clear_now_data()

    def pause(self):
        plt.ioff()
        plt.show()



