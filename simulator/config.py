#clean order file
FILE_NAME = "C:/Users/Jiyao/PycharmProjects/Simulator/data/Taxi_Trips_Chicago_08_2019_clean.csv"

#map
TOTAL_ZONES = 77

#time
#simulation step is per minute
TOTAL_MINUTES_ONE_DAY = 1440
YEAR = 2019
MONTH = 8
ALL_DAYS_IN_MONTH = 31
TOTAL_TIME_STEP_ONE_EPISODE = 44660 #1460 #(one day)
ON_MONITOR = True

#driver generator
LOW_BOUND = 5
HIGH_BOUND = 5

#rider
RIDER_NUM = None #45924 #(one day) #None means import all riders
PATIENCE_TIME = 20

#random
SEED = 100