# Import necessary libraries
import pandas as pd 

# Each data Load
df1 = pd.read_csv('.././data/yellow_tripdata_2015-01.csv')
df2 = pd.read_csv('.././data/yellow_tripdata_2016-01.csv')
df3 = pd.read_csv('.././data/yellow_tripdata_2016-02.csv')
df4 = pd.read_csv('.././data/yellow_tripdata_2016-03.csv')

# Set data list
df_list = [df1, df2, df3, df4]

# data cleaning
def data_cleaning(df):
    
    # Drop unnecessary columns
    df = df.drop(['store_and_fwd_flag'], axis=1).dropna()
    
    # Convert to datetime format
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

    # Calculate the time difference and convert to minutes
    df["trip_duration_minutes"] = (
        (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
    ).astype(float)
    
    # Remove cases where the trip duration is more than 1 hour or the distance is 0
    df = df[(df['trip_distance'] != 0) & (df['trip_duration_minutes'] < 60) & (df['trip_duration_minutes'] > 0)]
    
    # Remove the dropoff time
    df = df.drop(['tpep_dropoff_datetime'], axis=1)
    
    # Rename tpep_pickup_datetime column names
    df.rename(columns={"tpep_pickup_datetime": "TIME"}, inplace=True)
    
    return df

# Data transformation function
def data_change_db(df):
    
    # Data transformation
    df = df.melt(id_vars=['TIME'], var_name='name', value_name='value')

    # Convert 'TIME' column to dateTIME format
    df['TIME'] = pd.to_datetime(df['TIME'], format='mixed')

    # Change the order of the data columns
    df = df[['name', 'TIME', 'value']]
    df.columns = ['NAME', 'TIME', 'VALUE']

    # Convert epoch TIME to UTC TIME 
    epoch = pd.Timestamp('1970-01-01', tz='UTC')

    # Change TIME format
    df['TIME'] = pd.to_datetime(df['TIME'], format='%Y-%m-%d %H:%M:%S.%f')
    df['TIME'] = df['TIME'].dt.tz_localize('Asia/Seoul')
    df['TIME'] = df['TIME'].apply(lambda x: int((x - epoch).total_seconds() * 1_000_000_000))
    
    return df

# Save individual files
def convert(df_list):
    
    for i, df in enumerate(df_list):
        
        # Check number
        print(i)
        
        # Apply Each data cleaning
        df = data_cleaning(df)
        
        # Apply data transformation function (convert name, time, value format & epoch time)
        df = data_change_db(df)
        
        # Save the DataFrame to a CSV file
        df.to_csv(f'./datahub-2025-1-taxi_{i}.csv', index=False)
        
    print('done')
    
    
# convert
convert(df_list)

# Each data Load
df1 = pd.read_csv('./datahub-2025-1-taxi_0.csv')
df2 = pd.read_csv('./datahub-2025-1-taxi_1.csv')
df3 = pd.read_csv('./datahub-2025-1-taxi_2.csv')
df4 = pd.read_csv('./datahub-2025-1-taxi_3.csv') 

# Standardize the column names
df2.columns = df1.columns
df3.columns = df1.columns
df4.columns = df1.columns

# Concat each data
df = pd.concat([df1, df2, df3, df4], axis=0).reset_index(drop=True)

# Save total DataFrame to a CSV file
df.to_csv(f'./datahub-2025-1-taxi.csv', index=False)