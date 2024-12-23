import numpy as np
import pandas as pd
from scipy.fftpack import fft

# Hanning window function setup 
def set_hanning_window(sample_rate, df):
    
    # Generate Hanning window
    hanning_window = np.hanning(sample_rate)

    # Apply Hanning window to each row
    df_windowed = df.multiply(hanning_window, axis=1)
    
    return df_windowed

# FFT transformation function
def change_fft(sample_rate, df):
    # Total number of samples in the signal
    N = sample_rate
    
    fft_results = np.zeros((df.shape[0], N // 2 + 1), dtype=float)
    
    # Apply FFT to each row
    for i in range(df.shape[0]):
        
        # Calculate FFT for each row
        yf = fft(df.iloc[i].values)
        
        # Compute the absolute value of the FFT results and normalize (only the meaningful part)
        fft_results[i] = 2.0 / N * np.abs(yf[:N // 2 + 1])
    
    # Convert FFT results to a DataFrame
    fft_df = pd.DataFrame(fft_results)
    
    return fft_df

def FFT(sample_rate, df):
    
    # Apply hanning window
    df_hanning = set_hanning_window(sample_rate, df)
    
    # Apply FFT
    df_FFT = change_fft(sample_rate, df_hanning)
    
    return df_FFT
    