import numpy
import pandas as pd
from collections import defaultdict
import matplotlib.pylab as plt
from simulator.config import *
import numpy as np

FILE_NAME_Morning = "Taxi_Trips_Chicago_7AM_to_11AM_09_17_2019_clean.csv"
FILE_NAME_NOON = "Taxi_Trips_Chicago_11AM_to_3PM_09_17_2019_clean.csv"
FILE_NAME_EVENING = "Taxi_Trips_Chicago_5PM_to_9PM_09_17_2019_clean.csv"
TOTAL_TIME_STEP_ONE_EPISODE = 240
mean_range =15

def analyze_temporal_data():
    df = pd.read_csv(FILE_NAME_EVENING)
    demand = defaultdict(lambda : 0)
    plt.rcParams['figure.figsize'] = [8, 6]

    print("Importing Trip Data...")
    for i in range(len(df)):
        obs = df.iloc[i, :]

        #time
        timestamp = int(obs["Time"])
        demand[timestamp]+=1

    #time
    plt.figure(0)
    lists = sorted(demand.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    z = [0 for _ in range(len(y))]
    for i in range(len(y)):
        z[i] = sum(y[max(0,i-mean_range): min(len(y),i+mean_range+1)])/len(y[max(0,i-mean_range): min(len(y),i+mean_range+1)])

    #time
    plt.plot(x, y)
    plt.plot(x, z)
    #plt.xticks(range(0, 250, 60), ["7:00", "8:00", "9:00", "10:00", "11:00"], fontsize=16)
    #plt.xticks(range(0, 250, 60), ["11:00", "12:00", "13:00", "14:00", "15:00"], fontsize=16)
    plt.xticks(range(0, 250, 60), ["17:00", "18:00", "19:00", "20:00", "21:00"], fontsize=16)
    plt.yticks(range(10, 81, 10), ["10", "20", "30", "40", "50", "60", "70", "80"], fontsize=16)
    plt.xlabel("Time", fontsize=16)
    plt.ylabel("The Number of Tasks", fontsize=16)
    plt.show()



def analyze_spatial_data():
    df1 = pd.read_csv(FILE_NAME_Morning)
    df2 = pd.read_csv(FILE_NAME_NOON)
    df3 = pd.read_csv(FILE_NAME_EVENING)
    src1 = [0] * (TOTAL_ZONES + 1)
    src2 = [0] * (TOTAL_ZONES + 1)
    src3 = [0] * (TOTAL_ZONES + 1)
    plt.rcParams['figure.figsize'] = [8, 6]
    bar_width_c = 0.25
    bar_width_h = 0.3

    print("Importing Morning Trip Data...")
    for i in range(len(df1)):
        obs = df1.iloc[i, :]
        # spot src
        src1[int(obs["Pickup"])] += 1

    print("Importing Noon Trip Data...")
    for i in range(len(df2)):
        obs = df2.iloc[i, :]
        # spot src
        src2[int(obs["Pickup"])] += 1

    print("Importing Evening Trip Data...")
    for i in range(len(df3)):
        obs = df3.iloc[i, :]
        # spot src
        src3[int(obs["Pickup"])] += 1


    # src and dist
    x = list(range(1, 8)) + list(range(9, 28)) + list(range(29, 32)) + list(range(34, 76)) + [77]
    x = [str(i) for i in x]
    x1 = np.arange(len(x))
    x2 = [t+bar_width_c for t in x1]
    x3 = [t+bar_width_c for t in x2]
    y1 = np.array(src1[1:8] + src1[9:28] + src1[29:32] + src1[34:76] + src1[77:])
    y2 = np.array(src2[1:8] + src2[9:28] + src2[29:32] + src2[34:76] + src2[77:])
    y3 = np.array(src3[1:8] + src3[9:28] + src3[29:32] + src3[34:76] + src3[77:])
    bar_m = plt.bar(x1, y1, width=bar_width_c, label='Morning')
    bar_n = plt.bar(x2, y2, width=bar_width_c, label='Noon')
    bar_e = plt.bar(x3, y3, width=bar_width_c, label='Evening')
    '''plt.bar_label(bar_m)
    plt.bar_label(bar_n)
    plt.bar_label(bar_e)'''
    plt.xticks(x1+bar_width_c, x, fontsize=11)
    plt.yticks(range(50,351,50), fontsize=13)
    plt.xlabel("Zone ID", fontsize=20)
    plt.ylabel("Tasks Amount")
    plt.legend(fontsize=16)

    # copy from https://stackoverflow.com/questions/44863375/how-to-change-spacing-between-ticks
    plt.gca().margins(x=0)
    plt.gcf().canvas.draw()
    tl = plt.gca().get_xticklabels()
    maxsize = max([t.get_window_extent().width for t in tl])
    m = 0.2  # inch margin
    s = maxsize / plt.gcf().dpi * 77 + 2 * m
    margin = m / plt.gcf().get_size_inches()[0]
    plt.gcf().subplots_adjust(left=margin, right=1. - margin)
    plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()

    x = ['8', '28', '32', '33', '76']
    y1 = np.array(src1)[[8, 28, 32, 33, 76]]
    y2 = np.array(src2)[[8, 28, 32, 33, 76]]
    y3 = np.array(src3)[[8, 28, 32, 33, 76]]
    x1 = np.arange(len(x))
    x2 = [t + bar_width_h for t in x1]
    x3 = [t + bar_width_h for t in x2]
    bar_m = plt.bar(x1, y1, width=bar_width_h, label='Morning')
    bar_n = plt.bar(x2, y2, width=bar_width_h, label='Noon')
    bar_e = plt.bar(x3, y3, width=bar_width_h, label='Evening')
    plt.bar_label(bar_m, fontsize=16)
    plt.bar_label(bar_n, fontsize=16)
    plt.bar_label(bar_e, fontsize=16)
    plt.yticks(range(500, 5001, 500), fontsize=16)
    plt.xticks(x1 + bar_width_h, x, fontsize=16)
    plt.xlabel("Zone ID", fontsize=20)
    plt.ylabel("The Number of Tasks", fontsize=20)
    plt.legend(fontsize=16)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()




if __name__ == "__main__":
    analyze_spatial_data()