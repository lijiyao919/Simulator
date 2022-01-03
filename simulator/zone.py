import random
from driver import Driver
from rider import Rider

class Zone:
    __slots__ = ["_id", "_neighbor_zones", "_drivers_on_line", "_drivers_off_line",
                 "_riders_on_call", "_success_order_num", "_fail_order_num"]
    def __init__(self, zID):
        self._id = zID
        self._neighbor_zones = {}
        self._drivers_on_line = {}  #drivers that are available
        self._drivers_off_line = {}  #drivers in PICKUP, DROPOFF, and the zone is the destination
        self._riders_on_call = []
        self._success_order_num = 0
        self._fail_order_num = 0

    def __repr__(self):
        message = "cls:" + type(self).__name__ + ", id:" + str(self._id) + ", neighbor_zones:" + str(self._neighbor_zones.keys()) + \
                  ", drivers_online:" + str(self._drivers_on_line.keys()) + ", drivers_offline:" + str(self._drivers_off_line.keys()) + \
                  ", riders_oncall:" + str(self._riders_on_call) + ", success order number:" + str(self._success_order_num) + \
                  ", fail order num:" + str(self._fail_order_num)
        return message

    @property
    def id(self):
        return self._id

    @property
    def neighbod_zones(self):
        return self._neighbor_zones

    def add_neighbor_zones(self, zone):
        assert zone is not None
        assert zone.id is not self._id
        self._neighbor_zones[zone.id] = zone

    @property
    def drivers_on_line(self):
        return self._drivers_on_line

    def add_driver_on_line(self, driver):
        assert driver is not None
        assert driver.id not in self._drivers_off_line.keys()
        driver.zid = self._id
        self._drivers_on_line[driver.id] = driver

    def pop_driver_on_line_by_random(self):
        if len(self._drivers_on_line) != 0:
            return self._drivers_on_line.pop(random.choice(list(self._drivers_on_line.keys())))
        else:
            return None

    def pop_driver_on_line_by_id(self, driver_id):
        assert driver_id in self._drivers_on_line.keys()
        return self._drivers_on_line.pop(driver_id)

    @property
    def drivers_off_line(self):
        return self._drivers_off_line

    def add_driver_off_line(self, driver):
        assert driver is not None
        assert driver.id not in self._drivers_on_line.keys()
        driver.zid = self._id
        self._drivers_off_line[driver.id] = driver

    def pop_driver_off_line_by_id(self, driver_id):
        assert driver_id in self._drivers_off_line.keys()
        return self._drivers_off_line.pop(driver_id)

    @property
    def riders_on_call(self):
        return self._riders_on_call

    def add_riders(self, rider):
        assert rider.start_zone == self._id
        self._riders_on_call.append(rider)

    def pop_first_riders(self):
        if len(self._riders_on_call) != 0:
            return self._riders_on_call.pop(0)
        else:
            return None

    @property
    def success_order_num(self):
        return self._success_order_num

    def tick_success_order_num(self):
        self._success_order_num += 1

    @property
    def fail_order_num(self):
        return self._fail_order_num

    def tick_fail_order_num(self):
        self._fail_order_num += 1

if __name__ == "__main__":
    z1 = Zone(1)
    z2 = Zone(2)
    z3 = Zone(3)

    z1.add_neighbor_zones(z2)
    z1.add_neighbor_zones(z3)
    print(z1)
    print("neigbod zone id: "+str(z1.neighbod_zones[2].id))
    print("neigbod zone id: " + str(z1.neighbod_zones[3].id))

    driver1 = Driver(1, 30)
    driver2 = Driver(2, 30)

    z1.add_driver_on_line(driver1)
    z1.add_driver_on_line(driver2)
    print("driver online: ", z1.drivers_on_line.keys())
    driver = z1.pop_driver_on_line_by_random()
    print("driver online: ", z1.drivers_on_line.keys())

    z1.add_driver_off_line(driver)
    print("driver offline", z1.drivers_off_line.keys())
    driver = z1.pop_driver_off_line_by_id(driver.id)
    print("driver offline", z1.drivers_off_line.keys())
    z1.add_driver_on_line(driver)
    print("driver online: ", z1.drivers_on_line.keys())

    rider = Rider(1, 10, 1, 12, 40, 10, 20)
    z1.add_riders(rider)
    print("riders on call: ", z1.riders_on_call)
    z1.pop_first_riders()
    print("riders on call: ", z1.riders_on_call)

    z1.tick_success_order_num()
    print("success order num: "+str(z1.success_order_num))

    z1.tick_fail_order_num()
    print("fail order num: "+str(z1.fail_order_num))






