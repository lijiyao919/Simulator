import random
from driver import Driver
from rider import Rider

class Zone:
    def __init__(self, zID):
        self._id = zID
        self._neighbor_zones = {}
        self._drivers_on_line = {}  #drivers that are available
        self._drivers_off_line = {}  #drivers in PICKUP, DROPOFF, and the zone is the destination
        self._riders_on_call = {}
        self._success_order_num = 0
        self._fail_order_num = 0

    def __repr__(self):
        message = "cls:" + type(self).__name__ + ", id:" + str(self._id) + ", neighbor_zones:" + str(self._neighbor_zones.keys()) + \
                  ", drivers_online:" + str(self._drivers_on_line.keys()) + ", drivers_offline:" + str(self._drivers_off_line.keys()) + \
                  ", riders_oncall:" + str(self._riders_on_call.keys()) + ", success order number:" + str(self._success_order_num) + \
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

    def add_driver_on_line(self, driver):
        assert driver is not None
        assert driver.id not in self._drivers_off_line.keys()
        self._drivers_on_line[driver.id] = driver

    def pop_driver_on_line_by_random(self):
        if len(self._drivers_on_line) != 0:
            driver = self._drivers_on_line.pop(random.choice(list(self._drivers_on_line.keys())))
            return driver
        else:
            return None

    def add_driver_off_line(self, driver):
        assert driver is not None
        assert driver.id not in self._drivers_on_line.keys()
        self._drivers_off_line[driver.id] = driver

    def pop_driver_off_line_by_id(self, driver_id):
        assert driver_id in self._drivers_off_line.keys()
        driver = self._drivers_off_line.pop(driver_id)
        return driver

    def add_riders(self, rider):
        assert rider.status == self._id
        self._riders_on_call[rider.id] = rider

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
    print(z1)
    driver = z1.pop_driver_on_line_by_random()
    print(z1)

    z1.add_driver_off_line(driver)
    print(z1)

    driver = z1.pop_driver_off_line_by_id(driver.id)
    print(z1)
    z1.add_driver_on_line(driver)
    print(z1)

    z1.tick_success_order_num()
    print("success order num: "+str(z1.success_order_num))

    z1.tick_fail_order_num()
    print("fail order num: "+str(z1.fail_order_num))






