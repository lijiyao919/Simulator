import pandas as pd
import random

FILE_NAME = "Taxi_Trips_Chicago_5PM_to_9PM_09_17_2019.csv"
CLEAN_FILE_NAME = "Taxi_Trips_Chicago_5PM_to_9PM_09_17_2019_clean.csv"
YEAR = 2019
MONTH = 9
DATE = 17
TIME_HOUR = 17
TOTAL_TIME_STEP_ONE_EPISODE = 250

def clean_trips_data():
    df = pd.read_csv(FILE_NAME)
    d = {"ID": [],
         "Time": [],
         "Pickup": [],
         "Dropoff": [],
         "Duration": [],
         "Fare": [],
         }

    print("Importing Trip Data...")
    for i in range(len(df)):
        obs = df.iloc[i, :]

        # id
        actor_id = "R" + str(i)
        d["ID"].append(actor_id)

        #time
        timestamp = int((pd.to_datetime(obs["Trip Start Timestamp"]) - pd.Timestamp(YEAR, MONTH, DATE, TIME_HOUR)) / pd.to_timedelta(1, unit='m'))
        timestamp = timestamp + round(random.uniform(0, 15))
        assert 0<=timestamp<=TOTAL_TIME_STEP_ONE_EPISODE
        d["Time"].append(timestamp)

        # zone id
        pickup_zone = int(obs["Pickup Community Area"])
        dropoff_zone = int(obs["Dropoff Community Area"])
        assert 1<=pickup_zone<=77
        assert 1<=dropoff_zone<=77
        d["Pickup"].append(pickup_zone)
        d["Dropoff"].append(dropoff_zone)

        #Duration
        trip_duration_in_minute = 1 if int(float(obs["Trip Seconds"].replace(',', ''))/60)==0 else int(float(obs["Trip Seconds"].replace(',', ''))/60)
        assert trip_duration_in_minute != 0
        d["Duration"].append(trip_duration_in_minute)

        #Price
        if isinstance(obs["Trip Total"], float):
            d["Fare"].append(obs["Trip Total"])
        else:
            d["Fare"].append(float(obs["Trip Total"].replace(',', '')))

    print("Write data...")
    df = pd.DataFrame(d)
    df.sort_values(by="Time", ascending=True, inplace=True)
    df.to_csv(CLEAN_FILE_NAME, index=False)


if __name__ == "__main__":
    clean_trips_data()
    df1 = pd.read_csv(CLEAN_FILE_NAME)
    df2 = pd.read_csv(FILE_NAME)
    assert len(df1) == len(df2)
