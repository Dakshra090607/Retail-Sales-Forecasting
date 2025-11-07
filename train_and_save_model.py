import pandas as pd
from prophet import Prophet
import os
import logging
import warnings
from datetime import datetime

# Suppress Prophet warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

# --- Configuration ---
FILE_NAME = "stores_sales_forecasting.csv"
FORECAST_PERIOD = 365 # Forecast for the next 365 days
OUTPUT_FORECAST_FILE = "sales_forecast_results.csv"

# --- Data Loading and Preprocessing ---

def load_data(file_path):
    """Loads and preprocesses the sales data, handling encoding issues."""
    if not os.path.exists(file_path):
        logging.error(f"FATAL: Data file '{file_path}' not found.")
        return None

    try:
        # FIX: Explicitly set encoding to handle common CSV decoding errors
        df = pd.read_csv(file_path, encoding='latin-1') 
        logging.info("Data loaded successfully using latin-1 encoding.")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='cp1252')
            logging.warning("Used cp1252 encoding as latin-1 failed.")
        except Exception as e:
            logging.error(f"FATAL: Could not decode CSV file. Error: {e}")
            return None
    except Exception as e:
        logging.error(f"Unexpected data loading error: {e}")
        return None

    # Preprocessing
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df.dropna(subset=['Order Date', 'Sales'], inplace=True)
    
    # Prepare data for Prophet (ds, y)
    df_prophet = df.groupby('Order Date')['Sales'].sum().reset_index()
    df_prophet.rename(columns={'Order Date': 'ds', 'Sales': 'y'}, inplace=True)
    
    logging.info(f"Prophet training data ready. Shape: {df_prophet.shape}")
    return df_prophet

# --- Forecasting with Prophet ---

def prophet_forecast(df_prophet, periods):
    """Trains a Prophet model and forecasts future sales."""
    
    if df_prophet.shape[0] < 2:
        logging.warning("Insufficient data points to run Prophet. Need at least 2 distinct dates.")
        return None

    logging.info("Starting Prophet model training...")
    try:
        m = Prophet(
            yearly_seasonality=True, 
            weekly_seasonality=True, 
            daily_seasonality=False,
            # Adjust changepoint prior for slightly more flexibility in trend changes
            changepoint_prior_scale=0.05 
        )
        m.fit(df_prophet)

        # Create Future Dataframe and Predict
        future = m.make_future_dataframe(periods=periods)
        forecast = m.predict(future)
        
        logging.info("Forecasting completed successfully.")
        return forecast

    except Exception as e:
        logging.error(f"Prophet forecasting failed: {e}")
        return None

# --- Main Execution ---

if __name__ == "__main__":
    print(f"--- Sales Forecasting Model Trainer Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    
    # 1. Load Data
    data_for_prophet = load_data(FILE_NAME)
    
    if data_for_prophet is None:
        print("Training script aborted due to data loading errors.")
    else:
        # 2. Train Model and Generate Forecast
        forecast_results = prophet_forecast(data_for_prophet, FORECAST_PERIOD)
        
        if forecast_results is not None:
            # 3. Save Results
            # Include 'ds', 'yhat', and bounds for the dashboard
            final_forecast = forecast_results[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            final_forecast.to_csv(OUTPUT_FORECAST_FILE, index=False)
            
            print(f"\n✅ Forecast Complete! Results saved to: {OUTPUT_FORECAST_FILE}")
            print(f"Forecast period: {FORECAST_PERIOD} days.")
            print(f"Last date in forecast: {final_forecast['ds'].iloc[-1].strftime('%Y-%m-%d')}")
        else:
            print("\n❌ Forecasting failed or could not run. Check logs for details.")