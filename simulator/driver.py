from simulator.rider import Rider

class Driver:

    __slots__ = ["_id", "_zid", "_wake_up_time", "_rider", "_on_line", "_total_relocate_effort"]

    def __init__(self, wID, pos):
        self._id = wID
        self._zid = pos

        self._on_line = True
        self._wake_up_time = 0 # the time when driver wake up from offline to online
        self._rider = None
        self._total_relocate_effort = 0

    def __repr__(self):
        message = "cls:" + type(self).__name__ + ", id:" + str(self._id) +", wake_up_time:" + str(self._wake_up_time) + \
                  ", pos:" + str(self._zid) + ", on_line: " + str(self._on_line) + ", total_relocate_effort: " + str(self._total_relocate_effort)
        return message+", rider info: ["+str(self._rider)+"]" if self._rider is not None else message

    @property
    def id(self):
        return self._id

    @property
    def zid(self):
        return self._zid

    @zid.setter
    def zid(self, p):
        self._zid = p

    @property
    def on_line(self):
        return self._on_line

    @on_line.setter
    def on_line(self, s):
        self._on_line = s

    @property
    def wake_up_time(self):
        return self._wake_up_time

    @wake_up_time.setter
    def wake_up_time(self, time):
        self._wake_up_time = time

    @property
    def rider(self):
        return self._rider

    def pair_rider(self, rider):
        assert type(rider).__name__=="Rider"
        self._rider = rider

    def finish_rider(self):
        self._rider = None

    @property
    def total_relocate_effort(self):
        return self._total_relocate_effort

    def tick_relocate_effort(self):
        assert self._rider is None
        self._total_relocate_effort += 1

if __name__ == "__main__":
    driver = Driver(1, 30)
    print(driver)
    driver.tick_relocate_effort()
    print("total relocate effort: ", driver._total_relocate_effort)

    rider = Rider(1, 10, 23, 12, 40, 10, 20)
    driver.pair_rider(rider)
    print(driver)
    print("id: ", driver.id)
    driver.zid = 40
    print("pos: ", driver.zid)
    driver.wake_up_time = 60
    print("wake up time: ", driver.wake_up_time)







