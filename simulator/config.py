import logging

#clean order file
#FILE_NAME = "../../data/Taxi_Trips_Chicago_7AM_to_11AM_09_17_2019_clean.csv"
#FILE_NAME = "../../data/Taxi_Trips_Chicago_11AM_to_3PM_09_17_2019_clean.csv"
FILE_NAME = "../../data/Taxi_Trips_Chicago_5PM_to_9PM_09_17_2019_clean.csv"

#Images Folder
IMGS_FOLDER = "./img/"

#map
TOTAL_ZONES = 77

#time
#simulation step is per minute
TOTAL_MINUTES_ONE_DAY = 250
YEAR = 2019
MONTH = 9
DATE = 17
ALL_DAYS_IN_MONTH = 31
TOTAL_TIME_STEP_ONE_EPISODE = 250
ON_MONITOR = False

#driver generator
TOTAL_DRIVER_NUM = 1000
START_ZONE = 56
TIME_TO_NEIGHBOR = 1

#rider
RIDER_NUM = None #45924 #(one day) #None means import all riders
PATIENCE_TIME = 11

#random
SEED = 100

#log
LOG_FOLDER = "logfiles/"
LOG_LEVEL = logging.DEBUG
LOG_EP_LEVEL = 20
LOG_DRIVER_TRAJECTORY_LEN = 10

#for the test in Sep 23-29
#FILE_NAME = "C:/Users/Jiyao/PycharmProjects/Simulator/data/Taxi_Trips_Chicago_09_23_29_2019_clean.csv"
#MONTH = 9
#DATE = 23
#TOTAL_TIME_STEP_ONE_EPISODE = 10100 #(one week)