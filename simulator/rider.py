class Rider:

    __slots__ = ["_id", "_start_time", "_start_zone", "_end_zone", "_trip_duration", "_price",
                 "_give_up_time", "_status", "_call_taxi_duration", "_wait_pick_duration"]

    def __init__(self, uID, sT, sZ, eZ, d, p, pat):
        self._id = uID
        self._start_time = sT
        self._start_zone = sZ
        self._end_zone = eZ
        self._trip_duration = d
        self._price = p
        self._give_up_time = sT+pat

        self._call_taxi_duration = 0
        self._wait_pick_duration = 0

    def __repr__(self):
        return "cls:" + type(self).__name__ + ", id:" + str(self._id) +", give_up_time:" + str(self._give_up_time) + ", start_time:" + str(self._start_time) +\
               ", start_zone:" + str(self._start_zone) + ", end_zone:" + str(self._end_zone) + ", trip_duration:" + str(self._trip_duration) +\
               ", price:" + str(self._price) + ", patience:" + str(self._give_up_time) + ", call_taxi_duration:" + str(self._call_taxi_duration) + \
               ", wait_pick_duration:" + str(self._wait_pick_duration)

    @property
    def id(self):
        return self._id

    @property
    def start_time(self):
        return self._start_time

    @property
    def start_zone(self):
        return self._start_zone

    @property
    def end_zone(self):
        return self._end_zone

    @property
    def trip_duration(self):
        return self._trip_duration

    @property
    def price(self):
        return self._price

    @property
    def give_up_time(self):
        return self._give_up_time

    @property
    def call_taxi_duration(self):
        return self._call_taxi_duration

    def tick_call_taxi_duration(self):
        self._call_taxi_duration += 1

    def reset_call_taxi_duration(self):
        self._call_taxi_duration = 0

    @property
    def wait_pick_duration(self):
        return self._wait_pick_duration

    @wait_pick_duration.setter
    def wait_pick_duration(self, duration):
        self._wait_pick_duration = duration

    def reset_pick_duration(self, duration):
        self._wait_pick_duration = 0

if __name__ == "__main__":
    rider = Rider(1, 10, 23, 12, 40, 10, 20)
    print(rider)
    print("ID: "+str(rider.id))
    print("start time: " + str(rider.start_time))
    print("start zone: " + str(rider.start_zone))
    print("end zone: " + str(rider.end_zone))
    print("trip duration: "+str(rider.trip_duration))
    print("price: " + str(rider.price))
    print("give up time: "+str(rider.give_up_time))
    rider.tick_call_taxi_duration()
    print("call taxi duration: "+str(rider.call_taxi_duration))
    rider.reset_call_taxi_duration()
    print("call taxi duration: " + str(rider.call_taxi_duration))
    rider.wait_pick_duration =3
    print("wait pick duration: " + str(rider.wait_pick_duration))

