import pandas as pd
import random

FILE_NAME = "Taxi_Trips_Chicago_08_2019.csv"
CLEAN_FILE_NAME = "Taxi_Trips_Chicago_08_2019_clean.csv"
YEAR = 2019
MONTH = 8
DATE = 1

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
        timestamp = int((pd.to_datetime(obs["Trip Start Timestamp"]) - pd.Timestamp(YEAR, MONTH, DATE, 0)) / pd.to_timedelta(1, unit='m'))
        timestamp = timestamp + round(random.uniform(0, 15))
        d["Time"].append(timestamp)

        # zone id
        d["Pickup"].append(int(obs["Pickup Community Area"]))
        d["Dropoff"].append(int(obs["Dropoff Community Area"]))

        #Duration
        d["Duration"].append(int(float(obs["Trip Seconds"])/60))

        #Price
        d["Fare"].append(float(obs["Trip Total"]))

    print("Write data...")
    df = pd.DataFrame(d)
    df.sort_values(by="Time", ascending=True, inplace=True)
    df.to_csv(CLEAN_FILE_NAME, index=False)


if __name__ == "__main__":
    clean_trips_data()
    df1 = pd.read_csv(CLEAN_FILE_NAME)
    df2 = pd.read_csv(FILE_NAME)
    assert len(df1) == len(df2)
