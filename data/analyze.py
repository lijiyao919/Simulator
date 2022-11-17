import pandas as pd
from collections import defaultdict
import matplotlib.pylab as plt
import numpy as np

FILE_NAME = "Taxi_Trips_Chicago_08_19_7AM_9PM_2019_clean.csv"
TOTAL_TIME_STEP_ONE_EPISODE = 240
mean_range =15

def analyze_data():
    df = pd.read_csv(FILE_NAME)
    demand = defaultdict(lambda : 0)
    duration = defaultdict(lambda: 0)

    print("Importing Trip Data...")
    for i in range(len(df)):
        obs = df.iloc[i, :]

        #time
        timestamp = int(obs["Time"])
        demand[timestamp]+=1

        #Duration
        trip_duration_in_minute = int(obs["Duration"])
        duration[trip_duration_in_minute]+=1

    plt.figure(0)
    lists = sorted(demand.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    z = [0 for _ in range(len(y))]
    for i in range(len(y)):
        z[i] = sum(y[max(0,i-mean_range): min(len(y),i+mean_range+1)])/len(y[max(0,i-mean_range): min(len(y),i+mean_range+1)])

    plt.plot(x, y)
    plt.plot(x, z)
    plt.xticks(range(0, 850, 60))
    plt.xlabel("Time")
    plt.ylabel("The Number of Demand")
    plt.show()

    plt.figure(1)
    lists = sorted(duration.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    plt.bar(x, y)
    plt.show()




if __name__ == "__main__":
    analyze_data()