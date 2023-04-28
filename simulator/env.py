from data.graph import AdjList_Chicago
from simulator.objects import Trips
from simulator.zone import Zone
from simulator.driver import Driver
from simulator.timer import Timer
from simulator.config import *
from simulator.monitor import Monitor
from simulator.log import MyLogger
import random

random.seed(SEED)

class Env:
    def __init__(self):
        self._graph = {}
        self._trips = Trips()
        self._trips.read_trips_from_csv(row=RIDER_NUM)
        self._drivers_tracker = {}
        self._done = False
        self._info = None
        self._monitor = Monitor(self._graph)
        self._logger = MyLogger(__name__).get_logger()

    def reset(self):
        self._create_graph()
        self._add_drivers_on_line()
        self._trips.reset_index()
        Timer.tick_episode()
        Timer.reset_time_step()
        self._done = False
        if ON_MONITOR:
            self._monitor.reset_metrics()

    def pre_step(self):
        # put current timestamp trips to each zone
        self._add_riders()

        # iterate on call riders to find give ups
        self._iterate_riders_on_call_for_give_up()

        # iterate all off-line drivers in each zone
        self._iterate_drivers_off_line_for_wake_up()

        return self._state()

    def step(self, actions, V=None):
        # move on line drivers to seek
        self._iterate_drivers_on_line_for_move(actions)

        if ON_MONITOR:
            self._monitor.record_current_supply_demand()

        # match drivers and riders at each zone
        self._dispatch_drivers_for_riders(V)

        # iterate on call riders to update call time
        self._iterate_riders_on_call_for_update_call_time()

        # iterate idle driver to update idle time
        self._iterate_drivers_for_update_idle_time()

        Timer.tick_time_step()

        if ON_MONITOR:
            self._monitor.plot_metrics_by_time()
            self._monitor.clear_now_data()

        if Timer.get_time_step() == TOTAL_TIME_STEP_ONE_EPISODE:
            self._done = True
            if ON_MONITOR:
                self._monitor.pause()

        return self._state(), None, self._done, self._info

    def show_graph(self):
        message = "Graph:\n"+"{\n"
        for zid in self._graph.keys():
            message += str(zid)+": "+str(self._graph[zid].neighbod_zones.keys())+"\n"
        message+="}"
        return message

    def show_lost_riders_num_in_spatial(self):
        message = "Lost riders each zone: {"
        for zid in self._graph.keys():
            if self._graph[zid].fail_order_num_now > 0:
                message += str(zid) + ":"+str(self._graph[zid].fail_order_num_now)+";  "
            self._graph[zid].clear_now_data()
        message += "}"
        self._logger.debug("%s-%s: %s", Timer.get_episode(), Timer.get_time_step(), message)

    def show_lost_drivers_trajectory(self):
        message = "Lost drivers: {"
        cnt = 0
        for d in self._drivers_tracker.values():
            if len(d.trajectory) >= LOG_DRIVER_TRAJECTORY_LEN:
                assert d.in_service is False
                message += str(d.id)+":"+str(d.trajectory)+"; "
                cnt += 1
        message += "} " + str(cnt)
        self._logger.debug("%s-%s: %s", Timer.get_episode(), Timer.get_time_step(), message)


    def show_metrics_in_summary(self):
        message = "Metrics In Summary:\n"

        all_total_order_num = 0
        all_success_order_num = 0
        all_fail_order_num = 0
        all_rider_call_time = 0
        all_driver_relocate_effort = 0
        all_driver_idle_time = 0

        for zid in self._graph.keys():
            all_total_order_num += self._graph[zid].total_order_num
            all_success_order_num += self._graph[zid].success_order_num
            all_fail_order_num += self._graph[zid].fail_order_num
            all_rider_call_time += self._graph[zid].riders_call_time

        for d in self._drivers_tracker.values():
            all_driver_relocate_effort += d.total_relocate_effort
            all_driver_idle_time += d.total_idle_time
        message += "The current episode: "+str(Timer.get_episode())+"\n"
        message += "The current step: "+str(Timer.get_time_step())+"\n"
        message += "all total order num: "+str(all_total_order_num)+"\n"
        message += "all total driver num: " + str(len(self._drivers_tracker)) + "\n"
        message += "success rate: "+str(round(all_success_order_num/all_total_order_num,6)*100)+"%\n"
        message += "average rider call time: " + str(round(all_rider_call_time / all_success_order_num, 2)) + "\n"
        message += "average idle time before pick: " + str(round(all_driver_idle_time / all_success_order_num, 2)) + "\n"
        message += "average idle time per driver: " + str(round(all_driver_idle_time / len(self._drivers_tracker), 2)) + "\n"
        message += "average reposition times before pick: " + str(round(all_driver_relocate_effort / all_success_order_num, 2)) + "\n"
        message += "average reposition times per driver: " + str(round(all_driver_relocate_effort / len(self._drivers_tracker), 2)) + "\n"

        return message

    def get_drivers_length(self):
        return len(self._drivers_tracker)

    @property
    def monitor_drivers(self):
        return self._drivers_tracker

    def _create_graph(self):
        #inititial each node
        for zid in AdjList_Chicago.keys():
            self._graph[zid] = Zone(zid)

        #fill adj for each node
        for zid in AdjList_Chicago.keys():
            for adj_id in AdjList_Chicago[zid]:
                self._graph[zid].add_neighbor_zones(self._graph[adj_id])

    def _state(self):
        state = {"driver_locs":[0]*self.get_drivers_length(), "on_call_rider_num":[0]*(TOTAL_ZONES+1), "online_driver_num":[0]*(TOTAL_ZONES+1)}
        for did, driver in self._drivers_tracker.items():
            state["driver_locs"][did] = driver.zid
        for zid, zone in self._graph.items():
            state["on_call_rider_num"][zid] = len(self._graph[zid].riders_on_call)
            state["online_driver_num"][zid] = len(self._graph[zid].drivers_on_line)
        return state

    def _add_riders(self):
        while self._trips.is_valid() and self._trips.get_trip().start_time == Timer.get_time_step():
            rider = self._trips.pop_trip()
            rider.reset_call_taxi_duration()   # bug fix for unfinished request at the end of last iteration
            self._graph[rider.start_zone].add_riders(rider)

    def _add_drivers_on_line(self):
        id = 0
        for _ in range(TOTAL_DRIVER_NUM):
            d = Driver(id, START_ZONE)
            self._graph[START_ZONE].add_driver_on_line(d)
            self._drivers_tracker[id] = d
            id+=1

    def _iterate_riders_on_call_for_give_up(self):
        for zid in self._graph.keys():
            while len(self._graph[zid].riders_on_call) > 0:
                r = self._graph[zid].riders_on_call[0]
                if r.give_up_time == Timer.get_time_step():
                    self._graph[zid].pop_riders(give_up=True)
                    self._graph[zid].tick_fail_order_num()
                    r.reset_call_taxi_duration()
                else:
                    break

    def _iterate_riders_on_call_for_update_call_time(self):
        for zid in self._graph.keys():
            for r in self._graph[zid].riders_on_call:
                r.tick_call_taxi_duration()

    def _iterate_drivers_for_update_idle_time(self):
        for d in self._drivers_tracker.values():
            if not d.in_service:
                d.tick_idle_time()

    def _iterate_drivers_off_line_for_wake_up(self):
        for zid in self._graph.keys():
            for did, d in self._graph[zid].drivers_off_line.copy().items():
                assert d.zid == zid
                assert d.on_line is False
                assert d.wake_up_time >= Timer.get_time_step()
                if d.wake_up_time == Timer.get_time_step():
                    if d.rider is not None:           # not idle move
                        assert d.in_service is True
                        d.rider.reset_call_taxi_duration()
                        d.finish_rider()
                    self._graph[zid].drivers_off_line.pop(did)
                    self._graph[zid].add_driver_on_line(d)

    def _iterate_drivers_on_line_for_move(self, actions):
        for zid in self._graph.keys():
            for did, d in self._graph[zid].drivers_on_line.copy().items():
                assert d.zid == zid
                assert d.on_line is True
                assert d.rider is None
                assert d.in_service is False
                act = actions[did]
                assert act != -1
                d.reward_zid = d.zid  # trace the last step src
                d.record_trajectory(d.zid)
                if act >= len(self._graph[zid].neighbod_zones):
                    continue
                zid_to_go = list(self._graph[zid].neighbod_zones.keys())[act]
                d.tick_relocate_effort()
                d.wake_up_time = Timer.get_time_step() + TIME_TO_NEIGHBOR
                self._graph[zid].pop_driver_on_line_by_id(did)
                self._graph[zid_to_go].add_driver_off_line(d)

    def _dispatch_drivers_for_riders(self, V):
        self._info = {"fail_math_rate": [None] * (TOTAL_ZONES + 1)}
        for zid in self._graph.keys():
            if len(self._graph[zid].drivers_on_line) > 0:
                self._info["fail_math_rate"][zid] = len(self._graph[zid].drivers_on_line)
                while len(self._graph[zid].drivers_on_line) > 0 and len(self._graph[zid].riders_on_call) > 0:
                    rider = self._select_rider_2(zid, V)
                    driver = self._graph[zid].pop_driver_on_line_by_random()
                    assert driver.zid == zid
                    assert driver.on_line is True
                    assert driver.rider is None
                    assert driver.in_service is False
                    driver.clear_trajectory()
                    driver.pair_rider(rider)
                    driver.wake_up_time = Timer.get_time_step() + rider.trip_duration
                    driver.pickup_zid = zid
                    self._graph[rider.end_zone].add_driver_off_line(driver)
                    self._graph[zid].tick_success_order_num()
            if self._info["fail_math_rate"][zid] is not None:
                self._info["fail_math_rate"][zid] = round(len(self._graph[zid].drivers_on_line)/self._info["fail_math_rate"][zid], 1)

    def _select_rider_1(self, zid):
        rider = self._graph[zid].pop_riders()
        return rider

    def _select_rider_2(self, zid, V):
        scores = {}
        for i, r in enumerate(self._graph[zid].riders_on_call):
            scores[i] = V[r.end_zone]

        #select rider
        max_v = float("-inf")
        max_k = 0
        for k, v in scores.items():
            if v > max_v:
                max_v = v
                max_k = k
        rider = self._graph[zid].pop_riders(max_k)
        return rider






if __name__ == "__main__":
    env = Env()
    obs = env.reset()
    print(obs)
    #print(env.show_graph())
    #print(env.show_drivers_in_spatial())
    #for _ in range(22):
        #env.step(None)
        #print(env.show_riders_in_spatial())
        #print(env.show_drivers_in_spatial())


