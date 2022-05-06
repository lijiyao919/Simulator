import matplotlib.pyplot as plt
import numpy as np
from simulator.timer import Timer
from simulator.config import *
import os
import datetime

class Monitor:
    _graph = None
    _drivers_tracker = None
    _on_call_rider_total_num = []
    _available_driver_total_num = []
    _in_service_driver_total_num = []
    _service_rate_till_now = []
    _on_call_rider_num_by_zone = []
    _available_driver_num_by_zone = []

    @staticmethod
    def init(graph, drivers_tracker):
        plt.ion()
        Monitor._graph = graph
        Monitor._drivers_tracker = drivers_tracker

    @staticmethod
    def reset_by_time():
        Monitor._on_call_rider_total_num = []
        Monitor._available_driver_total_num = []
        Monitor._in_service_driver_total_num = []
        Monitor._service_rate_till_now = []
        folder_path = os.path.join(IMGS_FOLDER, 'time')
        os.makedirs(folder_path, exist_ok=True)
        path = os.path.join(folder_path, str(Timer.get_date(Timer.get_time_step())-1))
        plt.savefig(path)

    @staticmethod
    def reset_by_zone():
        Monitor._on_call_rider_num_by_zone = []
        Monitor._available_driver_num_by_zone = []
        folder_path = os.path.join(IMGS_FOLDER, 'zone')
        os.makedirs(folder_path, exist_ok=True)
        path = os.path.join(folder_path, str(Timer.get_time(Timer.get_time_step())))
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.savefig(path)

    @staticmethod
    def plot_supply_demand_by_time():
        plt.figure(1)
        plt.clf()
        Monitor._on_call_rider_total_num.append(Monitor._calc_on_call_rider_total_num())
        Monitor._available_driver_total_num.append(Monitor._calc_available_driver_total_num())
        Monitor._in_service_driver_total_num.append(Monitor._calc_in_service_driver_total_num())
        plt.subplot(211)
        plt.title("Date: " + str(Timer.get_date(Timer.get_time_step())))
        plt.ylabel('Number#')
        plt.plot(Monitor._on_call_rider_total_num, label="CALL_R#")
        plt.plot(Monitor._available_driver_total_num, label="AVAIL_D#")
        plt.plot(Monitor._in_service_driver_total_num, label="IN_SERVICE_D#")
        plt.legend()
        plt.subplot(212)
        plt.xlabel('Time')
        plt.ylabel('%')
        Monitor._service_rate_till_now.append(Monitor._calc_service_rate_till_now())
        plt.plot(Monitor._service_rate_till_now, label="SR_till_now")
        plt.legend()
        plt.pause(0.001)  # pause a bit so that plots are updated

    @staticmethod
    def plot_supply_demand_by_zone():
        plt.figure(2)
        plt.clf()
        zones = np.arange(1,78)
        Monitor._iterate_on_call_rider_by_zone()
        Monitor._iterate_available_driver_num_by_zone()
        plt.title("Date:" + str(Timer.get_date(Timer.get_time_step())) +"   "+'Time:' + str(datetime.timedelta(minutes=Timer.get_time(Timer.get_time_step()))))
        plt.xticks([i for i in range(1, 78, 2)])
        plt.xlabel('Zones')
        plt.ylabel('Number#')
        plt.bar(zones - 0.1, Monitor._on_call_rider_num_by_zone, width=0.3, label="CALL_R#")
        plt.bar(zones + 0.2, Monitor._available_driver_num_by_zone, width=0.3, label="AVAIL_D#")
        plt.legend()
        plt.pause(0.001)  # pause a bit so that plots are updated

    @staticmethod
    def _calc_on_call_rider_total_num():
        on_call_rider_num = 0
        for zid in Monitor._graph.keys():
            on_call_rider_num += len(Monitor._graph[zid].riders_on_call)
        return on_call_rider_num

    @staticmethod
    def _calc_available_driver_total_num():
        available_driver_num = 0
        for zid in Monitor._graph.keys():
            available_driver_num += len(Monitor._graph[zid].drivers_on_line)
        return available_driver_num

    @staticmethod
    def _calc_in_service_driver_total_num():
        not_available_driver_num = 0
        for d in Monitor._drivers_tracker.values():
            if d.in_service:
                not_available_driver_num += 1
        return not_available_driver_num

    @staticmethod
    def _calc_service_rate_till_now():
        total_success_num = 0
        total_num = 0

        for zid in Monitor._graph.keys():
            total_num += Monitor._graph[zid].total_order_num
            total_success_num += Monitor._graph[zid].success_order_num

        return (total_success_num / total_num) * 100

    @staticmethod
    def _iterate_on_call_rider_by_zone():
        Monitor._on_call_rider_num_by_zone = []
        for zid in Monitor._graph.keys():
            Monitor._on_call_rider_num_by_zone.append(len(Monitor._graph[zid].riders_on_call))

    @staticmethod
    def _iterate_available_driver_num_by_zone():
        Monitor._available_driver_num_by_zone = []
        for zid in Monitor._graph.keys():
            Monitor._available_driver_num_by_zone.append(len(Monitor._graph[zid].drivers_on_line))

    @staticmethod
    def _calc_curr_service_rate():
        pass







