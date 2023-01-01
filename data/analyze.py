import pandas as pd
from collections import defaultdict
import matplotlib.pylab as plt
import numpy as np
import random

FILE_NAME = "Taxi_Trips_Chicago_8_17_2019.csv"
TOTAL_TIME_STEP_ONE_EPISODE = 1440
mean_range =15

def analyze_data():
    df = pd.read_csv(FILE_NAME)
    demand = defaultdict(lambda : 0)

    print("Importing Trip Data...")
    for i in range(len(df)):
        obs = df.iloc[i, :]

        #time
        timestamp = int((pd.to_datetime(obs["Trip Start Timestamp"]) - pd.Timestamp(2019, 8, 17, 0)) / pd.to_timedelta(1, unit='m'))
        timestamp = timestamp + round(random.uniform(0, 15))
        demand[timestamp]+=1

    plt.figure(0)
    lists = sorted(demand.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    z = [0 for _ in range(len(y))]
    for i in range(len(y)):
        z[i] = sum(y[max(0,i-mean_range): min(len(y),i+mean_range+1)])/len(y[max(0,i-mean_range): min(len(y),i+mean_range+1)])

    plt.rc('font', size=12)
    #plt.plot(x, y)
    plt.plot(x, z)
    plt.xticks(range(0, 1441, 60), range(0, 25, 1), fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Time (hours)", fontsize=15)
    plt.ylabel("The Number of Demand", fontsize=15)
    plt.show()

if __name__ == "__main__":
    analyze_data()