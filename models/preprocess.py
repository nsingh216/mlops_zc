import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
# import seaborn as sns
import yaml

from pandas.tseries.holiday import USFederalHolidayCalendar as f_cal
from prefect import flow, task
from prefect.blocks.notifications import SendgridEmail
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

@task(retries=3, retry_delay_seconds=3)
def load_data_from_bigquery(project:str, file_name:str) -> None:
    """
    Read in data from GCP BQ tables and save into a CSV file for further preprocessing
    Saving also allows us to train the model without reconnecting to GCP each time.

    @param: project: GCP project id with access to the public dataset
    """


    # Read data from bigquery    
    QUERY = """
        SELECT *
         FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips` AS trips
         JOIN `bigquery-public-data.austin_bikeshare.bikeshare_stations` AS stations
           ON trips.start_station_name=stations.name
    """

    df = pd.read_gbq(
        QUERY, 
        project_id=project
    )

    label_encoder = LabelEncoder()
    df['bike_id_enc'] = label_encoder.fit_transform(df['bike_id'])

    # print(df.shape)
    save_data_file(df, file_name)

    return df


def save_data_file(df: pd.DataFrame, file:str) -> None:
    """
    Save off dataframe into a csv file in the data directory

    @param: df : pandas dataframe containing the data to be saved
    @param: file : name of csv file where the data will be saved to
    """
    
    path = os.path.dirname(os.path.dirname(__file__))
    file_path = f"{path}/data/{file}.csv"
    
    df.to_csv(file_path, index=False)


def get_outliers():
    """
    Calculate the IQR range for ride_duration
    """

    models_path = os.path.dirname(__file__)
    print(models_path)
    outlier_file = os.path.join(models_path, 'outliers.yml')
    print("outlier_file: ", outlier_file)
    if os.path.exists(outlier_file):
        with open(outlier_file, 'r') as f:
            outliers = yaml.safe_load(f)

        outlier_min = outliers['MIN']
        outlier_max = outliers['MAX']
    
    else:
        data_path = os.path.join(os.path.dirname(models_path), "data")
        print("data_path: ", data_path)
        df = pd.read_csv(os.path.join(data_path, 'bike_share_raw.csv'), low_memory=False)

        # calc outlier using inner quartile range:
        outlier_min = np.percentile(df['duration_minutes'], 25, method='midpoint')
        outlier_max = np.percentile(df['duration_minutes'], 75, method='midpoint')

        # save it to a file for future use
        yaml_dict = {'MIN' : int(outlier_min), 'MAX' : int(outlier_max)}

        with open(outlier_file, 'w') as f:
            yaml.safe_dump(yaml_dict, f)

    return outlier_min, outlier_max



def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out records outside the IQR from provided dataframe
    """
    min, max = get_outliers()
    df_wo_outliers = df[df['duration_minutes'].between(min, max)]

    return df_wo_outliers



def get_project_name():
    """
    Get the GCP project name, used to download public dataset
    """

    file_path = os.path.dirname(os.path.dirname(__file__))
    full_file_path = f"{file_path}/config.yaml"

    with open(full_file_path, 'r') as f:
        config = yaml.safe_load(f)

    project_name = config['GCP_PROJECT']

    return project_name


def preprocess(df: pd.DataFrame): # , encoder: LabelEncoder):

    # drop unused colums:
    # col_drop = ['Unnamed: 0.1', 'Unnamed: 0', 'trip_id', 'end_station_id', 'end_station_name', 'station_id']

    # encode subscriber types
    df['subscriber_type_is_local'] = np.where(df['subscriber_type'].str.lower().str.contains('local'), 1, 0)
    df['subscriber_type_is_student'] = np.where(df['subscriber_type'].str.lower().str.contains('student'), 1, 0)
    df['subscriber_type_is_youth'] = np.where(df['subscriber_type'].str.lower().str.contains('youth'), 1, 0)
    df['subscriber_type_is_single_trip'] = np.where(df['subscriber_type'].str.lower().str.contains('single'), 1, 0)

    # encode bike type
    df['bike_type_enc'] = np.where(df['bike_type'] == 'classic', 1, 0)

    # encode start date
    cal = f_cal()
    holidays = cal.holidays(start = '2014-01-01', end = '2023-08-01')
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['start_day_of_week'] = df['start_time'].dt.dayofweek
    df['start_day_is_weekend'] = df['start_day_of_week'].isin([5, 6]).astype('int') # sat/sun = 5, 6 day of week
    df['start_month'] = df['start_time'].dt.month
    df['start_hour'] = df['start_time'].dt.hour
    df['start_minute'] = df['start_time'].dt.minute
    df['start_day_is_holiday'] = df['start_time'].dt.date.isin(holidays).astype('int')

    # fill in Nan with 0
    fill_in_nan = ['footprint_length', 'footprint_width', 'start_station_id', 'end_station_id']
    df[fill_in_nan] = df[fill_in_nan].fillna(0)

    # # label encoder 
    # if encoder:
    #     # for test data, just transform
    #     df['bike_id'] = [x if x in encoder.classes_ else -1 for x in df['bike_id']]
    #     encoder.transform(df['bike_id'])
    # else:
    #     # for training data, fit & transform
    #     label_encoder = LabelEncoder()
    #     df['bike_id_enc'] = label_encoder.fit_transform(df['bike_id'])

    df_ex_outliers = remove_outliers(df) 

    cols_to_return = [
        'subscriber_type_is_local',
        'subscriber_type_is_student',
        'subscriber_type_is_youth',
        'subscriber_type_is_single_trip',
        'bike_type_enc',
        'start_day_of_week',
        'start_day_is_weekend',
        'start_month',
        'start_hour',
        'start_minute',
        'start_day_is_holiday',
        'bike_id_enc',
        'start_station_id',
        'end_station_id',
        'council_district',

        'duration_minutes'
    ]

    return df_ex_outliers[cols_to_return] #, label_encoder


#@task
def get_preprocessed_data():
    """
    
    """
    FULL_FILE_NAME = "bike_share_raw"
    PREDICT_COL = 'duration_minutes'
    path = os.path.dirname(os.path.dirname(__file__))

    training_data_file = "bike_share_cln_train.csv"
    test_data_file = "bike_share_cln_train.csv"

    train_df = pd.read_csv(f"{path}/data/{training_data_file}", low_memory=False)
    train_x = train_df.loc[:, train_df.columns != PREDICT_COL]
    train_y = train_df[PREDICT_COL]

    test_df = pd.read_csv(f"{path}/data/{test_data_file}", low_memory=False)
    test_x = test_df.loc[:, test_df.columns != PREDICT_COL]
    test_y = test_df[PREDICT_COL]

    return train_x, train_y, test_x, test_y, test_data_file, training_data_file


@flow(name="Preprocess bikeshare data")
def main():

    global path
    global PREDICT_COL

    REFRESH_EXPORT = False
    REFRESH_SPLIT = False
    FULL_FILE_NAME = "bike_share_raw"
    PREDICT_COL = 'duration_minutes'
    path = os.path.dirname(os.path.dirname(__file__))

    # 1. start by exporting the full dataset from BQ
    if REFRESH_EXPORT:
        project = get_project_name()
        full_df = load_data_from_bigquery(project, FULL_FILE_NAME)
    
    else:
        full_df = pd.read_csv(f"{path}/data/{FULL_FILE_NAME}.csv", low_memory=False)

    # 2. split out data into train & test - just randomly 30/70 split for now
    if REFRESH_SPLIT:
        X_train, X_test, y_train, y_test = train_test_split(
            full_df.loc[:, full_df.columns != PREDICT_COL], 
            full_df['duration_minutes'], 
            test_size=0.3,
            random_state=42
        )

        train_df = pd.concat([X_train, y_train], axis=1)
        save_data_file(train_df, f"{FULL_FILE_NAME}_train")
        test_df = pd.concat([X_test, y_test], axis=1)
        save_data_file(test_df, f"{FULL_FILE_NAME}_test")

    else:
        train_df = pd.read_csv(f"{path}/data/{FULL_FILE_NAME}_train.csv", low_memory=False)
        test_df = pd.read_csv(f"{path}/data/{FULL_FILE_NAME}_test.csv", low_memory=False)

        
        print(train_df.dtypes)
        print(train_df.head())

    # 3. preprocess the training data
    print(train_df.shape)  
    print(test_df.shape)

    train_proc_df = preprocess(train_df)

    # 4. preprocess the test set
    test_proc_df = preprocess(test_df)

    training_data_file = "bike_share_cln_train"
    test_data_file = "bike_share_cln_test"

    save_data_file(train_proc_df, training_data_file)
    save_data_file(test_proc_df, test_data_file)

    # sendgrid_block = SendgridEmail.load("BLOCK_NAME")
    # sendgrid_block.notify("Hello from Prefect!")
    



if __name__ == "__main__":
    main()