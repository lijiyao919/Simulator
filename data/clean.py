import pandas as pd
import datetime

FILE_NAME="Taxi_Trips_08_2019.csv"
CLEAN_FILE_NAME = "Chicago_08_29_clean.csv"
YEAR = 2019
MONTH = 8

def clean_trips_data():
    df = pd.read_csv(FILE_NAME)
    d = {"ID": [],
         "Time": [],
         "Weekday": [],
         "Pickup": [],
         "Dropoff": [],
         "Duration": [],
         "Fare": [],
         }
    cols = ["ID", "Time", "Weekday", "Pickup", "Dropoff", "Duration", "Fare"]

    print("Importing Trip Data...")
    for i in range(len(df)):
        obs = df.iloc[i, :]

        # id
        actor_id = "R" + str(i)
        d["ID"].append(actor_id)

        #time
        date = int(obs["Trip Start Timestamp"].split('/')[1])
        d["Time"].append(int((pd.to_datetime(obs["Trip Start Timestamp"]) - pd.Timestamp(YEAR, MONTH, date, 0)) / pd.to_timedelta(1, unit='m')))

        #weekday
        d["Weekday"].append(datetime.date(YEAR,MONTH,date).weekday())

        # zone id
        d["Pickup"].append(int(obs["Pickup Community Area"]))
        d["Dropoff"].append(int(obs["Dropoff Community Area"]))

        #Duration
        d["Duration"].append(int(float(obs["Trip Seconds"])/60))

        #Price
        d["Fare"].append(float(obs["Trip Total"]))

    print("Write data...")
    df = pd.DataFrame(d)
    df.to_csv(CLEAN_FILE_NAME, index=False)


if __name__ == "__main__":
    clean_trips_data()
    df1 = pd.read_csv(CLEAN_FILE_NAME)
    df2 = pd.read_csv(FILE_NAME)
    assert len(df1) == len(df2)
