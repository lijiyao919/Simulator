from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from simulator.rider import Rider
import math

FILE_NAME = "C:/Users/Jiyao/PycharmProjects/Simulator/data/Chicago_08_29_clean.csv"

class Distribution(ABC):
    @abstractmethod
    def sample(self):
        pass

class UniformDistribution(Distribution):
    def __init__(self, low, high):
        self._low = low
        self._high = high

    def sample(self, seed=0):
        np.random.seed(seed)
        return math.ceil(np.random.uniform(self._low, self._high))

class GaussianDistribution(Distribution):
    def __init__(self, mu, sigma):
        self._mu = mu
        self._sigma = sigma

    def sample(self, seed=0):
        np.random.seed(seed)
        return math.ceil(np.random.normal(self._mu, self._sigma))

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
    uf = UniformDistribution(1,10)
    print(uf.sample())
    normal = GaussianDistribution(100,10)
    print(normal.sample())

    it = Trips()
    it.read_trips_from_csv(row=7)
    print(it.length)


