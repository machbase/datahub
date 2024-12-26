import pandas as pd 
from urllib.parse import quote

# Data loading function
def data_load(URL, table, name, start_time, end_time, timeformat, resample_freq=None):
    
    # URL Encoding Time
    start_time_ = quote(start_time)
    end_time_ = quote(end_time)
    
    # Load data 
    df = pd.read_csv(f'{URL}/db/tql/datahub/api/v2/tql/select-rawdata.tql?table={table}&name={name}&start={start_time_}&end={end_time_}&timeformat={timeformat}')
        
    # Convert to data grouped by the time
    df = df.pivot_table(index='TIME', columns='NAME', values='VALUE', aggfunc='first').reset_index()
    
    # Set time index
    df.set_index('TIME', inplace=True)
    
    # Set datetime format
    df.index = pd.to_datetime(df.index)
    
    # Resample data only if resample_freq is provided
    if resample_freq:
        df = df.resample(resample_freq).mean()  # Mean aggregation for resampling

    return df    