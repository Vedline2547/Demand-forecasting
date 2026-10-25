# Demand Forecasting Using Time Series Models

## Project Overview

This project applies time series analysis and forecasting techniques to historical product sales data to understand demand patterns and predict future sales.

The project combines Exploratory Data Analysis (EDA), data preprocessing, time series visualization, and three forecasting models: **ARIMA, Prophet, and Exponential Smoothing**.

The models are evaluated using Root Mean Squared Error (RMSE) and forecast visualizations to compare their performance on weekly sales data.

Based on the numerical evaluation, **ARIMA achieved the lowest RMSE and performed best overall** among the three models. Prophet also showed promising visual performance when its forecasts were compared with actual demand.

## Project Objectives

- Explore historical sales data and understand demand patterns.
- Clean and prepare data for time series analysis.
- Identify trends, seasonality, volatility, and autocorrelation.
- Develop forecasting models using ARIMA, Prophet, and Exponential Smoothing.
- Evaluate model accuracy using RMSE.
- Compare forecast predictions against actual demand.
- Identify the best-performing model for the evaluated dataset.

## Dataset Description

The dataset contains weekly product sales records from different stores.

| Column | Description |
|---|---|
| `record_ID` | Sales record identifier |
| `week` | Date associated with the sales record |
| `store_id` | Store identifier |
| `sku_id` | Stock Keeping Unit (product) identifier |
| `total_price` | Total product price |
| `base_price` | Base product price |
| `is_featured_sku` | Indicates whether a product is featured |
| `is_display_sku` | Indicates whether a product is displayed |
| `units_sold` | Number of units sold |

For this project, sales are aggregated across products and stores by week to create an overall weekly demand time series.

## Exploratory Data Analysis (EDA)

Exploratory Data Analysis is performed in `main.py` to understand the structure and characteristics of the dataset before forecasting.

The analysis includes:

- Inspecting data types and identifying missing values.
- Handling missing `total_price` values using SKU-level median prices, followed by the overall median as a fallback.
- Converting the `week` column into datetime format.
- Aggregating `units_sold` by week.
- Visualizing sales distributions using histograms and boxplots.
- Examining relationships between numerical variables using a correlation heatmap.
- Plotting weekly sales to identify demand trends.
- Performing seasonal decomposition with a 52-week seasonal period.
- Calculating percentage changes to investigate demand volatility.
- Applying 4-week and 52-week moving averages.
- Analyzing autocorrelation to identify relationships between current and previous sales.

These techniques help reveal important characteristics of the time series and inform the forecasting process.

## Forecasting Models

### 1. ARIMA

ARIMA stands for AutoRegressive Integrated Moving Average. It models temporal dependencies in historical demand using autoregressive and moving-average components, with differencing when required.

**Implementation:**
- Uses `statsmodels.tsa.arima.model.ARIMA`.
- Implements an ARIMA(1, 0, 0) configuration.
- Splits the time series chronologically into 80% training data and 20% testing data.
- Generates forecasts for the test period.
- Evaluates predictions using RMSE.
- Visualizes the original time series and forecasts.

File: `arima_model.py`

### 2. Prophet

Prophet is a time series forecasting library designed to model temporal patterns, including trends and seasonality.

**Implementation:**
- Converts the data into Prophet's required `ds` and `y` columns.
- Uses an 80/20 chronological train-test split.
- Enables yearly seasonality.
- Generates forecasts for the test horizon.
- Calculates RMSE.
- Visualizes actual demand against predicted demand.

File: `prophet_model.py`

### 3. Exponential Smoothing

Exponential Smoothing forecasts demand by assigning greater weight to recent observations. Seasonal smoothing can help represent recurring demand patterns.

**Implementation:**
- Uses `statsmodels.tsa.holtwinters.ExponentialSmoothing`.
- Applies additive seasonality with a 52-week seasonal period.
- Uses an 80/20 chronological train-test split.
- Generates forecasts for the test period.
- Evaluates predictions using RMSE.
- Visualizes training data, actual test data, and forecasted values.

File: `smoothing.py`

## Model Evaluation and Results

The three models were evaluated using Root Mean Squared Error (RMSE), which measures the magnitude of prediction errors. A lower RMSE indicates better numerical forecasting accuracy when models are evaluated on the same observations.

### RMSE Comparison

| Rank | Model | RMSE |
|---:|---|---:|
| 1 | ARIMA | 6,235 |
| 2 | Prophet | 7,709 |
| 3 | Exponential Smoothing | 13,387 |

### Results Interpretation

**ARIMA was the best-performing model numerically**, achieving the lowest RMSE of 6,235.

Prophet recorded an RMSE of 7,709, while Exponential Smoothing recorded the highest RMSE of 13,387.

Compared with Prophet, ARIMA's RMSE was approximately 19.1% lower. Compared with Exponential Smoothing, ARIMA's RMSE was approximately 53.4% lower.

Although Prophet appeared to follow actual demand more closely in the forecast visualizations, ARIMA produced lower numerical forecasting error. This demonstrates the value of combining visual analysis with quantitative evaluation when selecting a forecasting model.

These results apply to the dataset, model configurations, and test period used in this project. For a fair comparison, all models should be evaluated against the same test dates and actual sales observations.

## Technologies and Libraries

- **Python** — implementation and analysis
- **Pandas** — data manipulation and preprocessing
- **NumPy** — numerical operations
- **Matplotlib** — time series and forecast visualizations
- **Seaborn** — statistical visualizations
- **Statsmodels** — ARIMA, Exponential Smoothing, and time series analysis
- **Prophet** — time series forecasting
- **Scikit-learn** — forecasting error evaluation

## Project Structure

```text
Demand-forecasting/
├── main.py
├── arima_model.py
├── prophet_model.py
├── smoothing.py
├── README.md
├── .gitignore
└── data (1).csv
```

The CSV file is shown for reference. If the dataset is not included in the repository, place it in the project directory before running the scripts.

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/Vedline2547/Demand-forecasting.git
cd Demand-forecasting
```

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install pandas numpy matplotlib seaborn statsmodels scikit-learn prophet
```

### 4. Run the scripts

Run the EDA script first, followed by the forecasting models:

```bash
python main.py
python arima_model.py
python prophet_model.py
python smoothing.py
```

Ensure the CSV file path in each script points to the correct location of your dataset.

## Limitations and Future Improvements

- The current forecasts model aggregated weekly sales rather than individual product or store demand.
- The ARIMA configuration has not been systematically tuned.
- The 52-week seasonal period requires sufficient historical observations to estimate annual seasonality reliably.
- Forecast dates and test observations should be aligned consistently across all models.
- Further evaluation using Mean Absolute Error (MAE) and residual diagnostics could provide additional insight into model performance.

Potential improvements include:

- Hyperparameter tuning for ARIMA and Exponential Smoothing.
- Rolling-origin time series cross-validation.
- Forecasting demand separately for each SKU or store.
- Incorporating prices, promotions, and display indicators as additional predictors.
- Comparing additional forecasting models and evaluation metrics.

## Conclusion

This project demonstrates an end-to-end time series forecasting workflow, from data preprocessing and exploratory analysis to forecasting, visualization, and model evaluation.

ARIMA, Prophet, and Exponential Smoothing were implemented to compare different approaches to demand forecasting. **ARIMA achieved the lowest RMSE of 6,235**, making it the best-performing model numerically in this evaluation, while Prophet showed promising visual alignment with actual demand.

The project provides practical experience in time series analysis, forecasting, model evaluation, and data visualization using Python.

## Author

**Vedline Ochieng**

GitHub: [@Vedline2547](https://github.com/Vedline2547)
