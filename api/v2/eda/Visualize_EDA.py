import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

# Make HeatMap
def HeatMap(df, x_size, y_size):
    
    ## Set path for saving model training results 
    os.makedirs('./EDA/HeatMap', exist_ok=True)
    
    # Calculate correlation coefficient
    corr = df.corr()

    # Remove columns that are entirely NaN
    filtered_corr = corr.dropna(how='all', axis=1).dropna(how='all', axis=0)

    # Identify removed columns
    removed_columns = set(df.columns) - set(filtered_corr.columns)

    # Print the result
    print("Removed NaN Columns:", removed_columns)

    # Display heatmap (based on data values)
    plt.figure(figsize=(x_size, y_size))
    sns.heatmap(filtered_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("HeatMap")

    # Save the heatmap
    plt.savefig('./EDA/HeatMap/heatmap.png')

    # Show the heatmap on the screen
    plt.show()
    
# Make Plot
def plot(df, x_size, y_size):
    
    ## Set path for saving model training results 
    os.makedirs('./EDA/plot', exist_ok=True)    
    
    # Iterate over each column and plot the data
    for column in df.columns:
        plt.figure(figsize=(x_size, y_size))
        plt.plot(df[column])
        plt.title(f'{column} Plot')
        plt.xlabel('Index')
        plt.ylabel(column)
        
        # Save the plot as a file
        plt.savefig(f'./EDA/plot/{column}_plot.png')
        
        # Display the plot
        plt.show()
        
# Make Decomposition
def Decomposition(df, period, x_size, y_size):
    
    ## Set path for saving model training results 
    os.makedirs('./EDA/decomposition', exist_ok=True)
    
    # Iterate over each column for decomposition
    for column in df.columns:
        if df[column].dtype == 'float64' or df[column].dtype == 'int64':  # Check if the column is numeric
            # Decompose the column using statsmodels
            decomposition = sm.tsa.seasonal_decompose(df[column], model='additive', period=period)  # Adjust 'period' based on data
            
            # Plot decomposition components
            plt.figure(figsize=(x_size, y_size))
            
            # Plot trend
            plt.subplot(411)
            plt.plot(decomposition.trend)
            plt.title(f'{column} Trend')
            
            # Plot seasonal
            plt.subplot(412)
            plt.plot(decomposition.seasonal)
            plt.title(f'{column} Seasonal')
            
            # Plot residual
            plt.subplot(413)
            plt.plot(decomposition.resid)
            plt.title(f'{column} Residual')
            
            # Plot original
            plt.subplot(414)
            plt.plot(df[column])
            plt.title(f'{column} Original')
            
            # Save the plots as a file
            plt.tight_layout()
            plt.savefig(f'./EDA/decomposition/{column}_decomposition.png')
            
            # Display the plot
            plt.show()
            
# Main Function
def Visualize_EDA(df, period, x_size, y_size, option):
    
    if option == 1:
        
        print('Start Visualize HeatMap')
        HeatMap(df, x_size, y_size)
        print('Save Finish') 
    
    if option == 2:
        
        print('Start Visualize For Each Column')
        plot(df, x_size, y_size)
        print('Save Finish') 
        
    if option == 3:
        
        print('Start Visualize Decomposition For Each Column')
        Decomposition(df, period, x_size, y_size)
        print('Save Finish') 
        
    if option == 4:
        
       print('Start Visualize All Method')
       HeatMap(df, x_size, y_size)
       plot(df, x_size, y_size)
       Decomposition(df, period, x_size, y_size)
       print('Save Finish')         
    
