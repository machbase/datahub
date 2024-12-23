import pandas as pd

# Function to display tag names
def show_column(IP, Table):
    
    # Load tag name data
    df = pd.read_csv(f'{IP}/db/tql/datahub/api/v2/tql/get-tag-names.tql?table={Table}')
    
    # Convert to list format
    df = df.values.reshape(-1)
    
    return df.tolist()