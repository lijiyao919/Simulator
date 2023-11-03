import pandas as pd
from collections import defaultdict
import matplotlib.pylab as plt
import numpy as np
import random

FILE_NAME = "Taxi_Trips_Chicago_08_29_2019.csv"
TOTAL_TIME_STEP_ONE_EPISODE = 1440
mean_range = 7

def analyze_data():
    df = pd.read_csv(FILE_NAME)
    demand_city = [0 for _ in range(1441)]
    demand_dt = [0 for _ in range(1441)]

    print("Importing Trip Data...")
    for i in range(len(df)):
        obs = df.iloc[i, :]

        #time
        timestamp = int((pd.to_datetime(obs["Trip Start Timestamp"]) - pd.Timestamp(2019, 8, 29, 0)) / pd.to_timedelta(1, unit='m'))
        timestamp = timestamp + round(random.uniform(0, 15))
        #if int(obs["Pickup Community Area"]) == 76: #airport
        if int(obs["Pickup Community Area"]) == 8 or int(obs["Pickup Community Area"]) == 28 or \
            int(obs["Pickup Community Area"]) == 32 or int(obs["Pickup Community Area"]) == 33: #downtown
                demand_dt[timestamp] += 1
        demand_city[timestamp]+=1 # city

    z_city = [0 for _ in range(0, 1440)]
    for i in range(0, 1440):
        #print(i, len(demand_city[max(0,i-mean_range): min(i+mean_range+1, 1440)]))
        z_city[i] = sum(demand_city[max(0,i-mean_range): min(i+mean_range+1,1440)]) / len(demand_city[max(0,i-mean_range): min(i+mean_range+1, 1440)])


    z_dt = [0 for _ in range(0, 1440)]
    for i in range(0, 1440):
        #print(i, len(demand_dt[max(0, i - mean_range): min(i + mean_range + 1, 1440)]))
        z_dt[i] = sum(demand_dt[max(0, i - mean_range): min(i + mean_range + 1, 1440)]) / len(demand_dt[max(0, i - mean_range): min(i + mean_range + 1, 1440)])

    x = range(0, 1440)

    plt.figure(0)
    plt.rc('font', size=12)
    plt.plot(x, z_city, linewidth=3, label="Whole City")
    plt.plot(x, z_dt, linewidth=3, label="Downtown")
    plt.xticks(range(0, 1441, 60), range(0, 25, 1), fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("Time (hours)", fontsize=22)
    plt.ylabel(f"The Number of Requests per Minute", fontsize=22)
    plt.legend()
    plt.grid(linestyle='--')
    plt.show()

    z_city = np.array(z_city)
    z_dt = np.array(z_dt)
    plt.plot(x, z_dt/z_city, linewidth=3)
    plt.xticks(range(0, 1441, 60), range(0, 25, 1), fontsize=18)
    plt.yticks(np.arange(0.4, 1, 0.1), range(40, 100, 10), fontsize=18)
    plt.xlabel("Time (hours)", fontsize=22)
    plt.ylabel("Percentage (%)", fontsize=22)
    plt.grid(linestyle='--')
    plt.show()

if __name__ == "__main__":
    analyze_data()