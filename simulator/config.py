#clean order file
FILE_NAME = "../../data/Taxi_Trips_Chicago_08_29_2019_clean.csv"

#Images Folder
IMGS_FOLDER = "./img/"

#map
TOTAL_ZONES = 77

#time
#simulation step is per minute
TOTAL_MINUTES_ONE_DAY = 1440
YEAR = 2019
MONTH = 8
DATE = 29
ALL_DAYS_IN_MONTH = 31
TOTAL_TIME_STEP_ONE_EPISODE = 1460 #(one day) #44660 #(one month)
ON_MONITOR = False

#driver generator
LOW_BOUND = 10
HIGH_BOUND = 10
TIME_TO_NEIGHBOR = 1
N_ACTIONS = 10

#rider
RIDER_NUM = None #45924 #(one day) #None means import all riders
PATIENCE_TIME = 7

#random
SEED = 100


#for the test in Sep 23-29
#FILE_NAME = "C:/Users/Jiyao/PycharmProjects/Simulator/data/Taxi_Trips_Chicago_09_23_29_2019_clean.csv"
#MONTH = 9
#DATE = 23
#TOTAL_TIME_STEP_ONE_EPISODE = 10100 #(one week)