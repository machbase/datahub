# Import necessary libraries
import pandas as pd 

# Load data 
df = pd.read_csv('./XAUUSD_2010-2023.csv')

# Data transformation function
def data_change_db(df):
    
    # Data transformation
    df = df.melt(id_vars=['time'], var_name='name', value_name='value')

    # Convert 'time' column to datetime format
    df['time'] = pd.to_datetime(df['time'], format='mixed')

    # Change the order of the data columns
    df = df[['name', 'time', 'value']]
    df.columns = ['NAME', 'TIME', 'VALUE']

    # Convert epoch time to UTC time 
    epoch = pd.Timestamp('1970-01-01', tz='UTC')

    # Change time format
    df['TIME'] = pd.to_datetime(df['TIME'], format='%Y-%m-%d %H:%M:%S.%f')
    df['TIME'] = df['TIME'].dt.tz_localize('Asia/Seoul')
    df['TIME'] = df['TIME'].apply(lambda x: int((x - epoch).total_seconds() * 1_000_000_000))
    
    return df

# Transform data
df = data_change_db(df)

# Save the DataFrame 
df.to_csv('./datahub-2024-12-Gold-Price.csv', index=False)