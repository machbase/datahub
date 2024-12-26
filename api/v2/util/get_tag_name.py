import pandas as pd

# Function to display tag names
def show_column(URL, Table):
    
    # Load tag name data
    df = pd.read_csv(f'{URL}/db/tql/datahub/api/v2/tql/get-tag-names.tql?table={Table}')
    
    # Convert to list format
    df = df.values.reshape(-1)
    
    return df.tolist()