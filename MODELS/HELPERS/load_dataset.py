import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def load_and_transform_data(trips_file):

    trips = pd.read_csv(trips_file)
    weathers = pd.read_csv(trips_file.replace("tripdata", "weatherdata"))

    # Convert to datetime
    trips["timestamp"] = pd.to_datetime(trips["timestamp"])
    weathers["timestamp"] = pd.to_datetime(weathers["timestamp"])
    
    # Set the 'Date' column as the index
    trips = trips.set_index("timestamp")
    weathers = weathers.set_index("timestamp")

    trips_np = trips.to_numpy()
    weathers_np = weathers.to_numpy()

    # One-hot encoding for season
    encoder = OneHotEncoder(sparse_output=False)
    weekends = trips.index.dayofweek.isin([5, 6])
    weekend_1hot = encoder.fit_transform(weekends.reshape(-1, 1))[:,1].reshape(-1,1)
    weekend_1hot = np.repeat(weekend_1hot[:, np.newaxis, :], trips.shape[1], axis=1)
    
    # Calculate the day in week
    day_in_week = trips.index.dayofweek.to_numpy()
    day_in_week_normalized = day_in_week / 7
    day_in_week_reshaped = np.repeat(
        day_in_week_normalized[:, np.newaxis], trips.shape[1], axis=1
    )

    # Calculate hour in day
    hours = trips.index.hour.to_numpy()
    hour_normalized = hours / 24
    hour_reshaped = np.repeat(hour_normalized[:, np.newaxis], trips.shape[1], axis=1)

    temperature = weathers_np[:, ::2]
    precipitation = weathers_np[:, 1::2]

    data = np.concatenate(
        (
            trips_np[:, :, np.newaxis],
            temperature[:, :, np.newaxis],
            precipitation[:, :, np.newaxis],
            hour_reshaped[:, :, np.newaxis],
            day_in_week_reshaped[:, :, np.newaxis],
        ),
        axis=2,
    )

    return data



def load_and_transform_data_OD(trips_file):

    trips = pd.read_csv(trips_file)

    # Convert to datetime
    trips["Date"] = pd.to_datetime(trips["Date"])
    
    # Set the 'Date' column as the index
    trips = trips.set_index("Date")

    trips_np = trips.iloc[:,1:].to_numpy()
    
     
    # Calculate the day in week
    day_in_week = trips.index.dayofweek.to_numpy()
    day_in_week_normalized = day_in_week / 7
    day_in_week_reshaped = np.repeat(
        day_in_week_normalized[:, np.newaxis], trips_np.shape[1], axis=1
    )

    # Calculate hour in day
    horaires = {
        '23:00-01:00': 0, '01:00-07:00': 1, '07:00-09:00': 2, '09:00-11:00': 3,
        '11:00-13:00': 4, '13:00-15:00': 5, '15:00-17:00': 6, '17:00-19:00': 7,
        '19:00-21:00': 8, '21:00-23:00': 9
    }
    horaire = trips["TrancheHoraire"].map(horaires).to_numpy()
    horaire_normalized = horaire / 10
    horaire_reshaped = np.repeat(horaire_normalized[:, np.newaxis], trips_np.shape[1], axis=1)
    
    # Convert dates to timestamps
    trips_date = np.array([date for date in trips.index])
    trips_date_reshaped = np.repeat(trips_date[:, np.newaxis], trips_np.shape[1], axis=1)

    data = np.concatenate(
        (
            trips_np[:, :, np.newaxis],
            horaire_reshaped[:, :, np.newaxis],
            day_in_week_reshaped[:, :, np.newaxis],
            trips_date_reshaped[:, :, np.newaxis],
        ),
        axis=2,
    )

    return data
