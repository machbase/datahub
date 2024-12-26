from urllib.parse import quote
import pandas as pd

# Function to calculate the maximum and minimum values for selected tag names
def set_minmax_value(URL, table, name, start_time_train, end_time_train):
    
    # URL encoding
    start = quote(start_time_train)
    end = quote(end_time_train)
    
    # Load Min, Max data
    df_ = pd.read_csv(f'{URL}/db/tql/datahub/api/v2/tql/select-scale.tql?table={table}&name={name}&start={start}&end={end}')
    
    # Set Min, Max values
    Min = df_.iloc[:,1:-1].T
    Max = df_.iloc[:,2:].T
    
    return Min, Max 