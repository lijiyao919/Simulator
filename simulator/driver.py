from rider import Rider

class Driver:
    IDLE = "idle"
    PICKUP = "pickup"
    DROPOFF = "dropoff"

    __slots__ = ["_id", "_pos", "_status", "_rider", "_total_relocate_effort"]

    def __init__(self, wID, pos):
        self._id = wID
        self._pos = pos

        self._status = Driver.IDLE
        self._rider = None
        self._total_relocate_effort = 0

    def __repr__(self):
        message = "cls:" + type(self).__name__ + ", id:" + str(self._id) +", status:" + str(self._status) + \
                  ", pos:" + str(self._pos) + ", total_relocate_effort: " + str(self._total_relocate_effort)
        return message+", rider info: ["+str(self._rider)+"]" if self._rider is not None else message

    @property
    def id(self):
        return self._id

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, p):
        self._pos = p

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, s):
        self._status = s

    @property
    def rider(self):
        return self._rider

    def pair_rider(self, rider):
        assert rider is not None
        self._rider = rider

    @property
    def total_relocate_effort(self):
        return self._total_relocate_effort

    def tick_relocate_effort(self):
        assert self._rider is None
        self._total_relocate_effort += 1

    def get_pick_duration(self):
        assert self._rider is not None
        return 1

    def get_trip_duration(self):
        assert self._rider is not None
        return self._rider.end_time - self._rider.start_time


if __name__ == "__main__":
    driver = Driver(1, 30)
    print(driver)
    driver.tick_relocate_effort()
    print("total relocate effort: ", driver._total_relocate_effort)

    rider = Rider(1, 10, 50, 23, 12, 10, 20)
    driver.pair_rider(rider)
    print(driver)
    print("id: ", driver.id)
    driver.pos = 40
    print("pos: ", driver.pos)
    driver.status = driver.PICKUP
    print("status: ", driver.status)
    print("pick duration: ", driver.get_pick_duration())
    print("trip duration: ", driver.get_trip_duration())







