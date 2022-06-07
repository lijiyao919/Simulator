import matplotlib.pyplot as plt
import numpy as np
from simulator.timer import Timer
from simulator.config import *
import os
import datetime

class Monitor:
    _graph = None
    _drivers_tracker = None
    _upcoming_rider_num_per_cycle_by_zone = []    #from zone dynamic
    _online_rider_num_per_cycle_by_zone = []
    _online_driver_num_per_cycle_by_zone = []
    _left_driver_num_per_cycle_by_zone = []
    _success_rider_num_per_cycle_by_zone = []             #from zone dynamic
    _lost_rider_num_per_cycle_by_zone = []                #from zone dynamic
    _value_function_per_cycle_by_zone = []
    _delta_per_cycle_by_zone = []

    _upcoming_rider_num_per_cycle_by_time = []
    _online_rider_num_per_cycle_by_time = []
    _left_driver_num_per_cycle_by_time = []
    _success_rate_rider_per_cycle_by_time = []
    _lost_rider_rate_per_cycle_by_time = []

    @staticmethod
    def init(graph, drivers_tracker):
        plt.ion()
        Monitor._graph = graph
        Monitor._drivers_tracker = drivers_tracker

    @staticmethod
    def plot_metrics_by_time():
        plt.figure(1)
        plt.clf()
        plt.subplot(411)
        plt.title("Date:" + str(Timer.get_date(Timer.get_time_step())) +"   "+'Time:' + str(datetime.timedelta(minutes=Timer.get_time(Timer.get_time_step()))))
        plt.ylabel('#')
        plt.plot(Monitor._upcoming_rider_num_per_cycle_by_time, label="UPCOMING_R#")
        plt.legend()
        plt.subplot(412)
        plt.ylabel('#')
        plt.plot(Monitor._online_rider_num_per_cycle_by_time, label="ONLINE_R#")
        plt.legend()
        plt.subplot(413)
        plt.ylabel('#')
        plt.plot(Monitor._left_driver_num_per_cycle_by_time, label="LEFT_D#")
        plt.legend()
        plt.subplot(414)
        plt.ylabel('%')
        plt.plot(Monitor._lost_rider_rate_per_cycle_by_time, label="LOST%")
        plt.xlabel('Time')
        plt.legend()
        plt.pause(0.001)  # pause a bit so that plots are updated
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

        folder_path = os.path.join(IMGS_FOLDER, 'time')
        os.makedirs(folder_path, exist_ok=True)
        path = os.path.join(folder_path, str(Timer.get_time_step()))
        plt.savefig(path)

        if Timer.get_time(Timer.get_time_step()) == TOTAL_MINUTES_ONE_DAY-1:
            Monitor._upcoming_rider_num_per_cycle_by_time = []
            Monitor._online_rider_num_per_cycle_by_time = []
            Monitor._left_driver_num_per_cycle_by_zone = []
            Monitor._lost_rider_rate_per_cycle_by_time = []



    @staticmethod
    def plot_metrics_by_zone():
        plt.figure(2)
        plt.clf()
        zones = np.arange(1,78)
        plt.subplot(411)
        plt.ylabel('Number#')
        plt.title("Date:" + str(Timer.get_date(Timer.get_time_step())) +"   "+'Time:' + str(datetime.timedelta(minutes=Timer.get_time(Timer.get_time_step()))))
        plt.xticks([i for i in range(1, 78, 2)])
        plt.bar(zones - 0.1, Monitor._online_rider_num_per_cycle_by_zone, width=0.3, label="CALL_R#")
        plt.bar(zones + 0.2, Monitor._online_driver_num_per_cycle_by_zone, width=0.3, label="AVAIL_D#")
        plt.legend()
        plt.subplot(412)
        plt.ylabel('Number#')
        plt.xticks([i for i in range(1, 78, 2)])
        plt.bar(zones - 0.1, Monitor._success_rider_num_per_cycle_by_zone, width=0.3, label="SUC_R#")
        plt.bar(zones + 0.2, Monitor._lost_rider_num_per_cycle_by_zone, width=0.3, label="LOST_R#")
        plt.legend()
        plt.subplot(413)
        plt.ylabel('Value Func')
        plt.xticks([i for i in range(1, 78, 2)])
        plt.bar(zones, Monitor._value_function_per_cycle_by_zone, width=0.3, label="Vs")
        plt.legend()
        plt.subplot(414)
        plt.ylabel('delta')
        plt.xlabel('Zones')
        plt.xticks([i for i in range(1, 78, 2)])
        plt.bar(zones, Monitor._delta_per_cycle_by_zone, width=0.3, label="delta")
        plt.legend()
        plt.pause(0.001)  # pause a bit so that plots are updated

        folder_path = os.path.join(IMGS_FOLDER, 'zone')
        os.makedirs(folder_path, exist_ok=True)
        path = os.path.join(folder_path, str(Timer.get_time_step()))
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.savefig(path)

    @staticmethod
    def collect_metrics_before_matching_from_env():
        Monitor._iterate_upcoming_rider_num_by_zone()
        Monitor._iterate_online_rider_by_zone()
        Monitor._iterate_online_driver_num_by_zone()

        Monitor._upcoming_rider_num_per_cycle_by_time.append(sum(Monitor._upcoming_rider_num_per_cycle_by_zone))
        Monitor._online_rider_num_per_cycle_by_time.append(sum(Monitor._online_rider_num_per_cycle_by_zone))

    @staticmethod
    def collect_metrics_after_matching_from_env():
        Monitor._iterate_success_rider_num_by_zone()
        Monitor._iterate_lost_rider_num_by_zone()
        Monitor._iterate_left_driver_num_by_zone()

        success_rate = (sum(Monitor._success_rider_num_per_cycle_by_zone) / sum(Monitor._online_rider_num_per_cycle_by_zone)) * 100 if sum(Monitor._online_rider_num_per_cycle_by_zone) != 0 else 0
        lost_rate = (sum(Monitor._lost_rider_num_per_cycle_by_zone) / sum(Monitor._online_rider_num_per_cycle_by_zone)) * 100 if sum(Monitor._online_rider_num_per_cycle_by_zone) != 0 else 0
        Monitor._success_rate_rider_per_cycle_by_time.append(success_rate)
        Monitor._lost_rider_rate_per_cycle_by_time.append(lost_rate)
        Monitor._left_driver_num_per_cycle_by_time.append(sum(Monitor._left_driver_num_per_cycle_by_zone))

    @staticmethod
    def collect_value_function_by_zone(Vs):
        Monitor._value_function_per_cycle_by_zone = Vs

    @staticmethod
    def collect_delta_by_zone(delta):
        Monitor._delta_per_cycle_by_zone = delta

    @staticmethod
    def _iterate_upcoming_rider_num_by_zone():
        Monitor._upcoming_rider_num_per_cycle_by_zone = []
        for zid in Monitor._graph.keys():
            Monitor._upcoming_rider_num_per_cycle_by_zone.append(Monitor._graph[zid].upcoming_order_num_per_cycle)

    @staticmethod
    def _iterate_online_rider_by_zone():
        Monitor._online_rider_num_per_cycle_by_zone = []
        for zid in Monitor._graph.keys():
            Monitor._online_rider_num_per_cycle_by_zone.append(len(Monitor._graph[zid].riders_on_call))

    @staticmethod
    def _iterate_online_driver_num_by_zone():
        Monitor._online_driver_num_per_cycle_by_zone = []
        for zid in Monitor._graph.keys():
            Monitor._online_driver_num_per_cycle_by_zone.append(len(Monitor._graph[zid].drivers_on_line))

    @staticmethod
    def _iterate_left_driver_num_by_zone():
        Monitor._left_driver_num_per_cycle_by_zone = []
        for zid in Monitor._graph.keys():
            Monitor._left_driver_num_per_cycle_by_zone.append(len(Monitor._graph[zid].drivers_on_line))

    @staticmethod
    def _iterate_success_rider_num_by_zone():
        Monitor._success_rider_num_per_cycle_by_zone = []
        for zid in Monitor._graph.keys():
            Monitor._success_rider_num_per_cycle_by_zone.append(Monitor._graph[zid].success_order_num_per_cycle)

    @staticmethod
    def _iterate_lost_rider_num_by_zone():
        Monitor._lost_rider_num_per_cycle_by_zone = []
        for zid in Monitor._graph.keys():
            Monitor._lost_rider_num_per_cycle_by_zone.append(Monitor._graph[zid].lost_order_num_per_cycle)







