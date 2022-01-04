from data.graph import AdjList_Chicago
from simulator.objects import Trips
from simulator.zone import Zone
from simulator.driver import Driver
from simulator.timer import Timer

#driver
LOW_BOUND = 10
HIGH_BOUND = 20

class Env:
    def __init__(self):
        self._graph = {}
        self._trips = Trips()
        self._trips.read_trips_from_csv(row=0)
        self._monitor_drivers = {}

    def reset(self):
        self._create_graph()
        self._add_drivers_on_line()
        self._trips.reset_index()
        Timer.reset_time()

    def step(self, actions):
        print("The current time stamp: ", Timer.get_time())

        #put current timestamp trips to each zone
        self._add_riders()

        #iterate on call riders to find give ups
        self._iterate_riders_on_call_for_give_up()

        #iterate all off-line drivers in each zone
        self._iterate_drivers_off_line_for_wake_up()

        #match drivers and riders at each zone
        self._dispatch_drivers_for_riders()

        #move on line drivers to seek
        self._iterate_drivers_on_line_for_move(actions)

        Timer.tick_time()

    def show_graph(self):
        message = "Graph:\n"+"{\n"
        for zid in self._graph.keys():
            message += str(zid)+": "+str(self._graph[zid].neighbod_zones.keys())+"\n"
        message+="}"
        return message

    def show_drivers_in_spatial(self):
        message = "Driver Dist:\n" + "{\n"
        for zid in self._graph.keys():
            message += str(zid) + ": driver_on_line: " + str(self._graph[zid].drivers_on_line.keys()) + "\n"
            message += "  : driver_off_line: " + str(self._graph[zid].drivers_off_line.keys()) + "\n"
        message += "}"
        return message

    def show_riders_in_spatial(self):
        message = "Riders Dist:\n" + "{\n"
        for zid in self._graph.keys():
            message += str(zid) + ": ["
            for r in self._graph[zid].riders_on_call:
                message += str(r.id)+ ", "
            message += "]\n"
        message += "}"
        return message

    def get_drivers_length(self):
        return len(self._monitor_drivers)

    def _create_graph(self):
        #inititial each node
        for zid in AdjList_Chicago.keys():
            self._graph[zid] = Zone(zid)

        #fill adj for each node
        for zid in AdjList_Chicago.keys():
            for adj_id in AdjList_Chicago[zid]:
                self._graph[zid].add_neighbor_zones(self._graph[adj_id])

    def _add_riders(self):
        while self._trips.is_valid() and self._trips.get_trip().start_time == Timer.get_time():
            rider = self._trips.pop_trip()
            self._graph[rider.start_zone].add_riders(rider)

    def _add_drivers_on_line(self):
        num_drivers = 1 #UniformDistribution(LOW_BOUND, HIGH_BOUND).sample()
        id = 0
        for zid in self._graph.keys():
            for _ in range(num_drivers):
                d = Driver(id, zid)
                self._graph[zid].add_driver_on_line(d)
                self._monitor_drivers[id] = d
                id+=1

    def _iterate_riders_on_call_for_give_up(self):
        for zid in self._graph.keys():
            while len(self._graph[zid].riders_on_call) > 0:
                r = self._graph[zid].riders_on_call[0]
                if r.give_up_time == Timer.get_time():
                    self._graph[zid].pop_first_riders()
                    self._graph[zid].tick_fail_order_num()
                else:
                    break

    def _iterate_drivers_off_line_for_wake_up(self):
        for zid in self._graph.keys():
            for did, d in self._graph[zid].drivers_off_line.copy().items():
                assert d.zid == zid
                if d.wake_up_time == Timer.get_time():
                    d.finish_rider()
                    self._graph[zid].drivers_off_line.pop(did)
                    self._graph[zid].add_driver_on_line(d)

    def _iterate_drivers_on_line_for_move(self, actions):
        for zid in self._graph.keys():
            for did, d in self._graph[zid].drivers_on_line.copy().items():
                assert d.zid == zid
                act = actions[did]
                if act >= len(self._graph[zid].neighbod_zones):
                    continue
                zid_to_go = list(self._graph[zid].neighbod_zones.keys())[act]
                d.tick_relocate_effort()
                self._graph[zid].pop_driver_on_line_by_id(did)
                self._graph[zid_to_go].add_driver_on_line(d)

    def _dispatch_drivers_for_riders(self):
        for zid in self._graph.keys():
            while len(self._graph[zid].drivers_on_line)>0 and len(self._graph[zid].riders_on_call)>0:
                rider = self._graph[zid].pop_first_riders()
                driver = self._graph[zid].pop_driver_on_line_by_random()
                driver.pair_rider(rider)
                driver.wake_up_time = Timer.get_time() + rider.trip_duration
                self._graph[rider.end_zone].add_driver_off_line(driver)
                self._graph[rider.end_zone].tick_success_order_num()




if __name__ == "__main__":
    env = Env()
    env.reset()
    print(env.show_graph())
    print(env.show_drivers_in_spatial())
    for _ in range(22):
        env.step(None)
        print(env.show_riders_in_spatial())
        print(env.show_drivers_in_spatial())

