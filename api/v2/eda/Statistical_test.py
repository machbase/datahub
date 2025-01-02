from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model
from arch.__future__ import reindexing
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import pandas as pd
import warnings
# ignore warnings
warnings.filterwarnings("ignore")

# Check Stationary

# ADF (Augmented Dickey-Fuller) Test
def adf_test(df):
    for column in df.columns:
        print(f"\nadf_test for column: {column}")
        series = df[column]
        # Check constant
        if series.nunique() == 1:
            print(f"Column '{column}' has constant values and will be skipped.")
            continue
        try:
            result = adfuller(series)
            print(f'ADF Statistic: {result[0]}')
            print(f'p-value: {result[1]}')
            print(f'Critical Values: {result[4]}')
            if result[1] < 0.05:
                print("The series is stationary")
                print('------------------------------------------------------------------------------------------------------------------')
            else:
                print("The series is not stationary")
                print('------------------------------------------------------------------------------------------------------------------')
        except ValueError as e:
            print(f"Error in column '{column}': {e}")        
        
# KPSS (Kwiatkowski-Phillips-Schmidt-Shin) Test
def kpss_test(df):
    for column in df.columns:
        print(f"\nkpss_test for column: {column}")
        series = df[column]
        # Check constant
        if series.nunique() == 1:
            print(f"Column '{column}' has constant values and will be skipped.")
            continue
        try:
            result = kpss(series, regression='c')
            print(f'KPSS Statistic: {result[0]}')
            print(f'p-value: {result[1]}')
            print(f'Critical Values: {result[3]}')
            if result[1] < 0.05:
                print("The series is not stationary")
                print('------------------------------------------------------------------------------------------------------------------')
            else:
                print("The series is stationary")
                print('------------------------------------------------------------------------------------------------------------------')
        except ValueError as e:
            print(f"Error in column '{column}': {e}")          
        
# PP (Phillips-Perron) Test
def pp_test(df):
    for column in df.columns:
        print(f"\npp_test for column: {column}")
        series = df[column]
        # Check constant
        if series.nunique() == 1:
            print(f"Column '{column}' has constant values and will be skipped.")
            continue
        try:
            # Perform the Phillips-Perron test
            pp_result = PhillipsPerron(series)
            # Print the test statistic and p-value
            print(f"\nPhillips-Perron Test for column: {column}")
            print(f"Test Statistic: {pp_result.stat}")
            print(f"p-value: {pp_result.pvalue}")
            # Assess stationarity
            if pp_result.pvalue < 0.05:
                print("The series is stationary")
                print('------------------------------------------------------------------------------------------------------------------')
            else:
                print("The series is not stationary")   
                print('------------------------------------------------------------------------------------------------------------------')
        except ValueError as e:
            print(f"Error in column '{column}': {e}")              

# Autocorrelation Test

# Ljung-Box Test      
def ljung_box_test(df, lags=10):
    for column in df.columns:
        print(f"\nLjung-Box Test for column: {column}")
        series = df[column]
        result = acorr_ljungbox(series, lags=lags)
        # Print the test statistic and p-value
        print(f"p-value: {result.lb_pvalue.iloc[-1]}")
        # Assess whether the series is white noise
        if result.lb_pvalue.iloc[-1] < 0.05:
            print("There is autocorrelation")
            print('------------------------------------------------------------------------------------------------------------------')
        else:
            print("No autocorrelation")
            print('------------------------------------------------------------------------------------------------------------------')          
        
# Heteroscedasticity Test

# Autoregressive Conditional Heteroskedasticity Test
def arch_test(df):
    # Train the ARCH model
    for column in df.columns:
        print(f"\narch_test for column: {column}")
        series = df[column]
        # Check constant
        if series.nunique() == 1:
            print(f"Column '{column}' has constant values and will be skipped.")
            continue
        try:
            model = arch_model(series, vol='ARCH', p=1)
            result = model.fit(disp='off')

            # Output model results
            print("ARCH Model Results:\n")
            
            # Output key parameters
            print(f"Omega (Intercept): {result.params['omega']:.2e}")
            print(f"Alpha[1] (ARCH Term): {result.params['alpha[1]']:.4f}")

            # Output model evaluation metrics
            print("\nModel Evaluation:")
            print(f"Log-Likelihood: {result.loglikelihood:.2f}")
            print(f"AIC (Akaike Information Criterion): {result.aic:.2f}")
            print(f"BIC (Bayesian Information Criterion): {result.bic:.2f}")

            # Add interpretation
            print("\nInterpretation:")
            if result.params['omega'] < 1e12:
                print("Omega value is low, indicating a low level of volatility in the model.")
            else:
                print("Omega value is high, indicating a higher level of volatility in the model.")

            if result.params['alpha[1]'] > 0.8:
                print("Alpha[1] is high, suggesting that past variances have a strong effect on current volatility.")
                print('------------------------------------------------------------------------------------------------------------------')
            else:
                print("Alpha[1] is low, suggesting that past variances have a weaker effect on current volatility.")
                print('------------------------------------------------------------------------------------------------------------------')
        except ValueError as e:
            print(f"Error in column '{column}': {e}")   

# Multicollinearity test             

# VIF (Variance Inflation Factor) Test
def vif_test(serise):
    serise = add_constant(serise)  # Add constant term for intercept
    vif_data = pd.DataFrame()
    vif_data["Variable"] = serise.columns
    vif_data["VIF"] = [variance_inflation_factor(serise.values, i) for i in range(serise.shape[1])]
    
    for index, row in vif_data.iterrows():
        variable = row['Variable']
        vif_value = row['VIF']
        if vif_value < 5:
            interpretation = "Low multicollinearity"
        elif 5 <= vif_value < 10:
            interpretation = "Moderate multicollinearity"
        else:
            interpretation = "High multicollinearity"
            
        print(f"Variable: {variable}, VIF: {vif_value:.2f}, Interpretation: {interpretation}")