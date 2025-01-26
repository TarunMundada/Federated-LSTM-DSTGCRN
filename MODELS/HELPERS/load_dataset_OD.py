import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def load_and_transform_data(trips_file):

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
