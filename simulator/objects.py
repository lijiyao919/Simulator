from abc import ABC, abstractmethod
import pandas as pd
from simulator.rider import Rider
from simulator.config import *

class Reward(ABC):
    @abstractmethod
    def reward_scheme(self, driver):
        pass

#from ICAART2022 paper
class Reward_ICAART(Reward):
    def reward_scheme(self, driver):
        if driver.in_service is True:
            return 2
        elif driver.on_line is True:
            return -2
        elif driver.on_line is False:
            return -1
        else:
            raise Exception("wrong reward")

class Trips:
    def __init__(self):
        self._trips=[]
        self._index = 0
        self._length = 0

    def read_trips_from_csv(self, row=None):
        df = pd.read_csv(FILE_NAME)
        if row is None:
            row = len(df)
        self._length = row

        print("Import Trips...")
        for i in range(row):
            obs = df.iloc[i, :]
            trip_id = obs["ID"]
            timestamp = int(obs["Time"])
            pickup_zone = int(obs["Pickup"])
            dropoff_zone = int(obs["Dropoff"])
            trip_duration = int(obs["Duration"])
            trip_fare = float(obs["Fare"])
            self._trips.append(Rider(trip_id, timestamp, pickup_zone, dropoff_zone, trip_duration, trip_fare, 20))
        print("Done.")

    def pop_trip(self):
        assert self._index < self._length
        r = self._trips[self._index]
        self._index += 1
        return r

    def get_trip(self):
        assert self._index < self._length
        return self._trips[self._index]

    def is_valid(self):
        return self._index < self._length

    def reset_index(self):
        self._index = 0

    @property
    def length(self):
        return self._length


if __name__ == "__main__":
    it = Trips()
    it.read_trips_from_csv(row=7)
    print(it.length)


