from rider import Rider

class Driver:
    IDLE = "idle"
    BUSY = "busy"

    def __init__(self, wID, pos):
        self._id = wID
        self._pos = pos

        self._status = Driver.IDLE
        self._rider = None
        self._total_relocate_effort = 0

    def __repr__(self):
        message = "cls:" + type(self).__name__ + ", id:" + str(self._id) +", status:" + str(self._status) + \
                  ", total_relocate_effort: " + str(self._total_relocate_effort)
        return message+", rider info: ["+str(self._rider)+"]" if self._rider is not None else message

    @property
    def id(self):
        return self._id

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, zone_id):
        self._pos = zone_id

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, sta):
        self._status = sta

    @property
    def rider(self):
        return self._rider

    @rider.setter
    def rider(self, rider):
        self._rider = rider

    @property
    def total_relocate_effort(self):
        return self._total_relocate_effort

    def tick_relocate_effort(self):
        self._total_relocate_effort += 1


if __name__ == "__main__":
    driver = Driver(1, 30)
    print(driver)
    rider = Rider(1, 10, 50, 23, 12, 10, 20)
    driver.rider = rider
    print(driver)
    print("id: ", driver.id)
    driver.pos = 40
    print("pos: ", driver.pos)
    driver.status = driver.BUSY
    print("status: ", driver.status)
    driver.tick_relocate_effort()
    print("total relocate effort: ", driver._total_relocate_effort)







