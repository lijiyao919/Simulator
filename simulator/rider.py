class Rider:
    CALL = "call"
    WAIT = "wait"
    ON_TRIP = "on_trip"
    FINISHED = "finished"
    CANCEL = "cancel"

    __slots__ = ["_id", "_start_time", "_end_time", "_start_zone", "_end_zone", "_price",
                 "_patience", "_status", "_call_taxi_duration", "_wait_pick_duration"]

    def __init__(self, uID, sT, eT, sZ, eZ, p, pat):
        self._id = uID
        self._start_time = sT
        self._end_time =eT
        self._start_zone = sZ
        self._end_zone = eZ
        self._price = p
        self._patience = pat

        self._status = Rider.CALL
        self._call_taxi_duration = 0
        self._wait_pick_duration = 0

    def __repr__(self):
        return "cls:" + type(self).__name__ + ", id:" + str(self._id) +", status:" + str(self._status) + ", start_time:"+ str(self._start_time)+\
               ", end_time:"+str(self._end_time) + ", start_zone:"+str(self._start_zone) + ", end_zone:" + str(self._end_zone) + \
               ", price:"+ str(self._price) + ", patience:" + str(self._patience) + ", call_taxi_duration:" + str(self._call_taxi_duration) + \
               ", wait_pick_duration:" + str(self._wait_pick_duration)

    @property
    def id(self):
        return self._id

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time

    @property
    def start_zone(self):
        return self._start_zone

    @property
    def end_zone(self):
        return self._end_zone

    @property
    def price(self):
        return self._price

    @property
    def patience(self):
        return self._patience

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, status):
        self._status = status

    @property
    def call_taxi_duration(self):
        return self._call_taxi_duration

    def tick_call_taxi_duration(self):
        self._call_taxi_duration += 1

    @property
    def wait_pick_duration(self):
        return self._wait_pick_duration

    @wait_pick_duration.setter
    def wait_pick_duration(self, duration):
        self._wait_pick_duration = duration


if __name__ == "__main__":
    rider = Rider(1, 10, 50, 23, 12, 10, 20)
    print(rider)
    print("ID: "+str(rider.id))
    rider.status = Rider.WAIT
    print("status: " + str(rider.status))
    print("start time: " + str(rider.start_time))
    print("end time: " + str(rider.end_time))
    print("start zone: " + str(rider.start_zone))
    print("end zone: " + str(rider.end_zone))
    print("price: " + str(rider.price))
    print("patience: "+str(rider.patience))
    rider.tick_call_taxi_duration()
    print("call taxi duration: "+str(rider.call_taxi_duration))
    rider.wait_pick_duration =3
    print("wait pick duration: " + str(rider.wait_pick_duration))

