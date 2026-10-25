import pandas as pd
from prophet import Prophet
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
# Prepare data for prophet
prophet_data = weekly_data.reset_index()
prophet_data.columns = ['ds','y']
# Split into training set
train_data_prophet = prophet_data[:int(0.8 * (len(prophet_data)))]
test_data_prophet = prophet_data[int(0.8 * len(prophet_data)):]
test_data = weekly_data['units_sold'][int(0.8 * (len(weekly_data))):]
# Fit a prophet model
prophet_model = Prophet(yearly_seasonality=True)
prophet_model.fit(train_data_prophet)
# Generate future dates for the test period
future = prophet_model.make_future_dataframe(periods=len(test_data_prophet),freq='W')
# Generate forecasts
prophet_predictions = prophet_model.predict(future)
# Extract predictions for the test period
forecast_values = prophet_predictions['yhat'].tail(len(test_data_prophet)).to_numpy()
# Extract actual sales values
actual_values = test_data_prophet['y'].to_numpy()
# Evaluate model using RMSE
prophet_rmse = sqrt(mean_squared_error(actual_values, forecast_values))
print("Prophet RMSE:", prophet_rmse)
plt.figure(figsize=(12, 6))
plt.plot(test_data_prophet['ds'],actual_values,label='Actual Demand')
plt.plot(test_data_prophet['ds'],forecast_values,label='Prophet Forecast',linestyle='--')
plt.title('Prophet Demand Forecast vs Actual Demand')
plt.xlabel('Week')
plt.ylabel('Units Sold')
plt.legend()
plt.tight_layout()
plt.show()

