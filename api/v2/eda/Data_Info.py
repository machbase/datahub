import pandas as pd
from urllib import parse
import requests 
import io

# Data loading function
def Data_Info(URL, table, timeformat, resample_freq=None):
    
    params = parse.urlencode({
                            "q": f"select * from {table}",
                            "format": "csv",
                            "timeformat": f"{timeformat}",
                            "compress": "true"
                            })
    
    # Load the data in compressed format
    df = requests.request("GET", f"{URL}/db/query", params=params, stream=True)

    # Load as CSV format
    df = pd.read_csv(io.BytesIO(df.content))
        
    # Convert to data grouped by the time
    df = df.pivot_table(index='TIME', columns='NAME', values='VALUE', aggfunc='first').reset_index()
    
    # Set time index
    df.set_index('TIME', inplace=True)
    
    # Set datetime format
    df.index = pd.to_datetime(df.index)
    
    # Resample data only if resample_freq is provided
    if resample_freq:
        df = df.resample(resample_freq).mean()  # Mean aggregation for resampling
        
    # Check tag types
    print(f'Tag List: {list(df.columns)}')
    print('--------------------------------------------')
    # Number of data
    print(f'Number of data: {df.shape[0]}')
    print('--------------------------------------------')
    # Data time range
    print(f'Start Time: {df.index[0]}')
    print(f'End Time: {df.index[-1]}')
    print('--------------------------------------------')
    # Check for missing values
    print(f'Missing Values: {df.isnull().sum().sum()}')
    print('--------------------------------------------')
    # Print data time intervals
    # Calculate the time intervals between data points
    time_diff = df.index.to_series().diff().dropna()

    # Calculate the smallest and largest intervals
    min_interval = time_diff.dt.total_seconds().min()
    max_interval = time_diff.dt.total_seconds().max()

    # Print time intervals
    print(f"Smallest time interval: {min_interval:.2f} seconds")
    print(f"Largest time interval: {max_interval:.2f} seconds")
    print('--------------------------------------------')

    # Calculate the sampling rate (using the smallest interval for maximum sampling rate)
    sampling_rate = 1 / min_interval
    print(f"Maximum sampling rate (per second): {sampling_rate:.2f} Hz")
    print('--------------------------------------------')