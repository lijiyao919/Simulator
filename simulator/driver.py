from simulator.rider import Rider

class Driver:

    __slots__ = ["_id", "_zid", "_wake_up_time", "_rider", "_on_line", "_in_service", "_total_relocate_effort", "_total_idle_time", "_reward_zid", "_pickup_zid", "_trajectory"]

    def __init__(self, wID, pos):
        self._id = wID
        self._zid = pos
        self._reward_zid = pos
        self._pickup_zid = None
        self._trajectory = []

        self._on_line = True   # available or not, offline means unavailable
        self._in_service = False # delivering riders or not
        self._wake_up_time = 0 # the time when driver wake up from offline to online
        self._rider = None
        self._total_relocate_effort = 0
        self._total_idle_time = 0

    def __repr__(self):
        message = "cls:" + type(self).__name__ + ", id:" + str(self._id) +", wake_up_time:" + str(self._wake_up_time) + \
                  ", pos:" + str(self.zid) + ", reward_pos:" + str(self.reward_zid) + ", on_line: " + str(self._on_line) + ", in_service: " + str(self._in_service) + \
                  ", total_relocate_effort: " + str(self._total_relocate_effort)
        return message+", rider info: ["+str(self._rider)+"]" if self._rider is not None else message

    @property
    def id(self):
        return self._id

    #current zone id
    @property
    def zid(self):
        return self._zid

    @zid.setter
    def zid(self, p):
        self._zid = p

    #last step zone id on move
    @property
    def reward_zid(self):
        return self._reward_zid

    @reward_zid.setter
    def reward_zid(self, p):
        self._reward_zid = p

    # zone id for marking pick up
    @property
    def pickup_zid(self):
        return self._pickup_zid

    @pickup_zid.setter
    def pickup_zid(self, p):
        self._pickup_zid = p

    @property
    def on_line(self):
        return self._on_line

    @on_line.setter
    def on_line(self, s):
        self._on_line = s

    @property
    def in_service(self):
        return self._in_service

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
        self._in_service = True

    def finish_rider(self):
        self._rider = None
        self._in_service = False

    @property
    def total_relocate_effort(self):
        return self._total_relocate_effort

    def tick_relocate_effort(self):
        assert self._rider is None
        self._total_relocate_effort += 1

    @property
    def total_idle_time(self):
        return self._total_idle_time

    def tick_idle_time(self):
        assert self._rider is None
        self._total_idle_time += 1

    @property
    def trajectory(self):
        return self._trajectory

    def record_trajectory(self, zid):
        self._trajectory.append(zid)

    def clear_trajectory(self):
        self._trajectory = []

if __name__ == "__main__":
    driver = Driver(1, 30)
    print(driver)
    driver.tick_relocate_effort()
    print("total relocate effort: ", driver.total_relocate_effort)
    driver.tick_idle_time()
    print("total idle time: ", driver.total_idle_time)


    rider = Rider(1, 10, 23, 12, 40, 10, 20)
    driver.pair_rider(rider)
    print(driver)
    print("id: ", driver.id)
    driver.zid = 40
    print("pos: ", driver.zid)
    driver.wake_up_time = 60
    print("wake up time: ", driver.wake_up_time)
    driver.finish_rider()
    print(driver)
    driver.reward_zid = 50
    print("reward pos: ", driver.reward_zid)

    driver.record_trajectory(8)
    driver.record_trajectory(10)
    print(driver.trajectory)
    driver.clear_trajectory()
    print(driver.trajectory)







