import pandas as pd
from collections import defaultdict
import matplotlib.pylab as plt
from simulator.config import *
import numpy as np

FILE_NAME = "Taxi_Trips_Chicago_7AM_to_11AM_09_17_2019_clean.csv"
TOTAL_TIME_STEP_ONE_EPISODE = 240
mean_range =15

def analyze_data():
    df = pd.read_csv(FILE_NAME)
    demand = defaultdict(lambda : 0)
    duration = defaultdict(lambda: 0)
    spatio = defaultdict(lambda : [0]*(TOTAL_ZONES+1))

    print("Importing Trip Data...")
    for i in range(len(df)):
        obs = df.iloc[i, :]

        #time
        timestamp = int(obs["Time"])
        demand[timestamp]+=1

        #Duration
        trip_duration_in_minute = int(obs["Duration"])
        duration[trip_duration_in_minute]+=1

        #spatio
        spatio[int(timestamp/10+1)][int(obs["Pickup"])] += 1

    #time
    plt.figure(0)
    lists = sorted(demand.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    z = [0 for _ in range(len(y))]
    for i in range(len(y)):
        z[i] = sum(y[max(0,i-mean_range): min(len(y),i+mean_range+1)])/len(y[max(0,i-mean_range): min(len(y),i+mean_range+1)])

    plt.plot(x, y)
    plt.plot(x, z)
    plt.xticks(range(0, 250, 60))
    plt.xlabel("Time")
    plt.ylabel("The Number of Demand")
    plt.show()

    #duration
    plt.figure(1)
    lists = sorted(duration.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    plt.bar(x, y)
    plt.show()

    #spatio
    for t, dist in spatio.items():
        plt.figure(2)
        plt.title(t*10)
        plt.bar(range(TOTAL_ZONES+1), dist)
        plt.xticks(range(0, TOTAL_ZONES+1, 2))
        plt.xlabel("Zone")
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()




if __name__ == "__main__":
    analyze_data()