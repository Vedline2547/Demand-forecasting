# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import autocorrelation_plot
# Load dataset
data = pd.read_csv(r"C:\Users\HP\Desktop\New ML\Demand-forecasting\data (1).csv")
# Display first few rows
print(data.head())
# Checking for missing values
missing_values = data.isnull().sum()
print(missing_values)
# Fill the missing value in 'total_price' with the median of 'total_price' for the corresponding 'sku_id'
# First,calculate the median total_price for each sku_id
median_price_per_sku = data.groupby('sku_id')['total_price'].median()
# Fill missing value
data['total_price'].fillna(data['sku_id'].map(median_price_per_sku),inplace=True)
# Fill any remaining missing prices using the overall median
data['total_price'] = data['total_price'].fillna(data['total_price'].median())
# Check if there is any missing values
print(data.isnull().sum())
# Convert week column to datetime format
data['week'] = pd.to_datetime(data['week'],format='%d/%m/%y')
# Aggregate data by week
weekly_data = data.groupby('week')['units_sold'].sum().reset_index()
# Display first few rows of aggregated data
print(weekly_data.head())
# Plot histograms for categorical columns and target variable
categorical_cols = ['store_id','sku_id','is_featured_sku','is_display_sku','units_sold']
for col in categorical_cols:
    plt.figure(figsize=(10,4))
    plt.hist(data[col],bins=30)
    plt.title(f'Distribution of {col}')
    plt.show()
# Boxplot for numerical columns
numerical_cols = ['total_price','base_price']
for col in numerical_cols:
    plt.figure(figsize=(10,4))
    sns.boxplot(x=data[col])
    plt.title(f'Boxplot of {col}')
    plt.show()
# Check correlation between features
correlation = data.corr()
# Plot a heatmap of the correlation matrix
plt.figure(figsize=(10,8))
sns.heatmap(correlation,annot=True,cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()
# Plot time series of units sold
plt.figure(figsize=(15,6))
plt.plot(weekly_data['week'],weekly_data['units_sold'])
plt.title('Time series of Units sold')
plt.xlabel('Week')
plt.ylabel('Units Sold')
plt.show()
# No clear trend or seasonality now doing decomposition
# Set 'week' as index of data
weekly_data.set_index('week',inplace=True)
# Decompose the time series with a period of 52 weeks
decomposition = seasonal_decompose(weekly_data['units_sold'],period=52)
# Plot  the trend,seasonality and residuals
plt.figure(figsize=(12,8))
# Trend
plt.subplot(411)
plt.plot(decomposition.trend,label='Trend')
# Seasonality
plt.subplot(412)
plt.plot(decomposition.seasonal,label='Seasonality')
plt.legend(loc='best')
# Residuals
plt.subplot(413)
plt.plot(decomposition.resid,label='Residuals')
plt.legend(loc='best')
# Original
plt.subplot(414)
plt.plot(weekly_data['units_sold'],label='Original')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
# VOLATILITY TEST
# Calculate the percentage change in 'units sold' from one week to next
weekly_data['units_sold_pct_change'] = weekly_data['units_sold'].pct_change()
# Calculate the standard deviation of the percetage changes
volatility = weekly_data['units_sold_pct_change'].std()
print(volatility)
# Calculate the 4-week and 52-week moving averages
weekly_data['4_week_MA'] = weekly_data['units_sold'].rolling(window=4).mean()
weekly_data['52_week_MA'] = weekly_data['units_sold'].rolling(window=52).mean()
# Plot the original time series and moving averages
plt.figure(figsize=(15,6))
plt.plot(weekly_data['units_sold'],label='Original')
plt.plot(weekly_data['4_week_MA'],label='4_week Moving Average')
plt.plot(weekly_data['52_week_MA'],label='52_week Moving Average')
plt.title('Time Series of Units Sold with Moving Averages')
plt.xlabel('Week')
plt.ylabel('Units sold')
plt.legend()
plt.show()
# Plot autocorrelation
plt.figure(figsize=(12,5))
autocorrelation_plot(weekly_data['units_sold'])
plt.show()
# Presence of seasonality from the autocorrelation_plot