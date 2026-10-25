import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from math import sqrt
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
# Load dataset
data = pd.read_csv(r"C:\Users\HP\Desktop\New ML\Demand-forecasting\data (1).csv")
# Split data into training and test sets
# Checking for missing values
missing_values = data.isnull().sum()
print(missing_values)
# Fill the missing value in 'total_price' with the median of 'total_price' for the corresponding 'sku_id'
# First,calculate the median total_price for each sku_id
median_price_per_sku = data.groupby('sku_id')['total_price'].median()
# Fill missing value
data['total_price'] = data['total_price'].fillna(data['sku_id'].map(median_price_per_sku))
# Fill any remaining missing prices using the overall median
data['total_price'] = data['total_price'].fillna(data['total_price'].median())
# Check if there is any missing values
print(data.isnull().sum())
# Convert week column to datetime format
data['week'] = pd.to_datetime(data['week'],format='%d/%m/%y')
# Aggregate sales by week
weekly_data = (data.groupby('week')['units_sold'].sum().sort_index().to_frame())
# Ensure the index is datetime-based
weekly_data.index = pd.to_datetime(weekly_data.index)
# Display first few rows of aggregated data
print(weekly_data.head())
# Resample to weekly frequency
weekly_data_resampled = (weekly_data['units_sold'].resample('W').sum())
print(weekly_data_resampled.head())
# Split the data chronologically: 80% training, 20% testing
split_index = int(0.8 * len(weekly_data_resampled))
train_data_resampled = weekly_data_resampled.iloc[:split_index]
test_data_resampled = weekly_data_resampled.iloc[split_index:]
# Fit an ARIMA model
arima_model = ARIMA(train_data_resampled,order=(1,0,0)).fit()
# Generate predictions for ARIMA model
arima_predictions  = arima_model.predict(start=test_data_resampled.index[0],end=test_data_resampled.index[-1])
# Calculating RMSE
arima_rmse = sqrt(mean_squared_error(test_data_resampled,arima_predictions))
print(arima_rmse)
# Plot original data and predictions
plt.figure(figsize=(10,6))
plt.plot(weekly_data,label='Original')
plt.plot(arima_predictions,label='ARIMA')
plt.legend()
plt.show()
