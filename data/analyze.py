import pandas as pd
import random
from collections import defaultdict
import matplotlib.pylab as plt

FILE_NAME = "Taxi_Trips_Chicago_5PM_to_9PM_09_17_2019_clean.csv"
YEAR = 2019
MONTH = 9
DATE = 17
TIME_HOUR = 17
TOTAL_TIME_STEP_ONE_EPISODE = 250

def analyze_data():
    df = pd.read_csv(FILE_NAME)
    demand = defaultdict(lambda : 0)
    duration = defaultdict(lambda: 0)

    print("Importing Trip Data...")
    for i in range(len(df)):
        obs = df.iloc[i, :]

        #time
        timestamp = int(obs["Time"])
        assert 0<=timestamp<=TOTAL_TIME_STEP_ONE_EPISODE
        demand[timestamp]+=1

        #Duration
        trip_duration_in_minute = int(obs["Duration"])
        assert trip_duration_in_minute != 0
        duration[trip_duration_in_minute]+=1

    plt.figure(0)
    lists = sorted(demand.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(x, y)
    plt.show()

    plt.figure(1)
    lists = sorted(duration.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    plt.bar(x, y)
    plt.show()




if __name__ == "__main__":
    analyze_data()