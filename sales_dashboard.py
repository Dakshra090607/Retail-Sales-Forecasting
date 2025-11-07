import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go 
import os
import logging
import numpy as np 

logging.basicConfig(level=logging.INFO)

# --- Configuration ---
FILE_NAME = "stores_sales_forecasting.csv"
FORECAST_RESULTS_FILE = "sales_forecast_results.csv"

# Set page configuration
st.set_page_config(
    page_title="Retail Sales & Forecasting Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Data Loading (Historical and Forecast) ---

@st.cache_data
def load_historical_data(file_path):
    """Loads and preprocesses the sales data."""
    if not os.path.exists(file_path):
        st.error(f"FATAL ERROR: The historical data file '{file_path}' was not found.")
        st.stop()

    try:
        # FIX: Explicitly setting 'latin-1' encoding
        df = pd.read_csv(file_path, encoding='latin-1')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='cp1252')
        except Exception as e:
            st.error(f"ERROR: Could not decode the CSV file. Please check file encoding.")
            st.stop()
    
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df.dropna(subset=['Order Date', 'Sales'], inplace=True)
    return df

@st.cache_data
def load_forecast_data(file_path):
    """Loads the pre-generated forecast results."""
    if not os.path.exists(file_path):
        st.warning(f"Forecast results file '{file_path}' not found. Please run 'train_and_save_model.py' first.")
        return None

    forecast_df = pd.read_csv(file_path)
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
    logging.info("Forecast data loaded successfully.")
    return forecast_df

# --- Visualization Functions (Now safely accept and use keys) ---

def plot_sales_over_time(df, key_suffix=""):
    """Plots the trend of total sales over time. Uses unique key."""
    
    df_sales_time = df.groupby('Order Date')['Sales'].sum().reset_index()
    df_sales_time.columns = ['Date', 'Total Sales']
    
    fig = px.line(
        df_sales_time,
        x='Date',
        y='Total Sales',
        title='Total Sales Trend by Order Date',
        template='plotly_white'
    )
    fig.update_layout(height=400)
    # FIX: Uses unique key composed of function name and suffix
    st.plotly_chart(fig, width='stretch', key=f'sales_time_plot_{key_suffix}')

def plot_sales_by_state(df, top_n=10, key_suffix=""):
    """Plots total sales for the top N states. Uses unique key."""
    
    df_sales_state = df.groupby('State')['Sales'].sum().reset_index()
    df_sales_state = df_sales_state.sort_values(by='Sales', ascending=False).head(top_n)
    
    fig = px.bar(
        df_sales_state,
        x='State',
        y='Sales',
        title=f'Top {top_n} States by Sales Volume',
        color='Sales',
        template='plotly_white'
    )
    fig.update_layout(height=400)
    # FIX: Uses unique key composed of function name and suffix
    st.plotly_chart(fig, width='stretch', key=f'sales_state_plot_{key_suffix}')

def plot_forecast_data(historical_df, forecast_df):
    """Visualizes the pre-calculated forecast and components."""
    
    df_actuals = historical_df.groupby('Order Date')['Sales'].sum().reset_index()
    df_actuals.rename(columns={'Order Date': 'ds', 'Sales': 'y'}, inplace=True)
    
    # --- Plot 1: Combined Actuals and Prediction ---
    
    st.markdown("##### Sales Forecast with Uncertainty Interval")
    
    forecast_plot_data = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    
    fig = px.line(
        df_actuals,
        x='ds',
        y='y',
        labels={'y': 'Actual Sales', 'ds': 'Date'},
        template='plotly_white'
    )
    
    # Add the predicted forecast line
    fig.add_scatter(
        x=forecast_plot_data['ds'], 
        y=forecast_plot_data['yhat'], 
        mode='lines', 
        name='Predicted Sales (yhat)',
        line=dict(color='#E50914') # Distinct color
    )
    
    # Add the uncertainty band
    fig.add_trace(
        go.Scatter(
            x=forecast_plot_data['ds'].tolist() + forecast_plot_data['ds'].tolist()[::-1],
            y=forecast_plot_data['yhat_upper'].tolist() + forecast_plot_data['yhat_lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(229, 9, 20, 0.15)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            name='Uncertainty Interval'
        )
    )
    
    # Adjust range and add unique key
    start_date = df_actuals['ds'].min()
    end_date = forecast_plot_data['ds'].max()
    fig.update_layout(xaxis_range=[start_date, end_date], showlegend=True, height=500)
    st.plotly_chart(fig, width='stretch', key='forecast_plot_main')
    
    # --- Plot 2: Raw Forecast Data Table ---
    st.markdown(f"##### Raw Future Forecast Data")
    last_historical_date = df_actuals['ds'].max()
    forecast_data_future = forecast_df[forecast_df['ds'] > last_historical_date]
    
    st.dataframe(
        forecast_data_future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        .set_index('ds')
        .rename(columns={'yhat': 'Predicted Sales', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'})
    )


# --- KPI Metric Generation and Summary ---

def calculate_kpis(historical_df, forecast_df):
    """Calculates key performance indicators."""
    total_sales = historical_df['Sales'].sum()
    total_transactions = historical_df.shape[0]
    
    # Get total sales for the last 365 days 
    last_date = historical_df['Order Date'].max()
    start_date_year_ago = last_date - pd.Timedelta(days=365)
    recent_data = historical_df[historical_df['Order Date'] >= start_date_year_ago]
    recent_sales = recent_data['Sales'].sum()
    
    forecast_sales = None
    sales_growth_percent = None

    if forecast_df is not None:
        # Calculate next year's forecast total based on the saved forecast file
        last_pred_date = forecast_df['ds'].max()
        first_pred_date = last_pred_date - pd.Timedelta(days=365)
        future_forecast = forecast_df[
            (forecast_df['ds'] >= first_pred_date) & 
            (forecast_df['ds'] <= last_pred_date)
        ]
        
        if not recent_data.empty and not future_forecast.empty:
            forecast_sales = future_forecast['yhat'].sum()
            sales_growth_percent = (forecast_sales - recent_sales) / recent_sales * 100

    return {
        'total_sales': total_sales,
        'total_transactions': total_transactions,
        'recent_sales': recent_sales,
        'forecast_sales': forecast_sales,
        'sales_growth_percent': sales_growth_percent
    }
    
def executive_summary_tab(kpis, historical_df):
    st.header("Executive Summary: Key Performance Indicators")
    
    # 1. KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    # Helper functions for formatting
    fmt_sales = lambda x: f"${x:,.0f}" if x is not None else "N/A"
    fmt_percent = lambda x: f"{x:+.1f} %" if x is not None else "N/A"
    
    col1.metric(
        label="Total Historical Sales", 
        value=fmt_sales(kpis['total_sales']), 
    )
    
    col2.metric(
        label="Total Transactions", 
        value=f"{kpis['total_transactions']:,.0f}"
    )

    if kpis['forecast_sales'] is not None:
        col3.metric(
            label="Last Year's Sales (Actuals)", 
            value=fmt_sales(kpis['recent_sales'])
        )
        col4.metric(
            label="Forecasted Sales (Next Year)",
            value=fmt_sales(kpis['forecast_sales']),
            delta=fmt_percent(kpis['sales_growth_percent'])
        )
    else:
        col3.metric(label="Recent Sales (Last Year)", value=fmt_sales(kpis['recent_sales']))
        col4.warning("Run Model for Forecast KPIs")

    st.markdown("---")
    
    # 2. Key Insights Plots (Side-by-side)
    st.subheader("High-Level Performance Distribution")
    
    plot_col1, plot_col2 = st.columns([1.5, 1])
    
    with plot_col1:
        st.caption("Sales Trend")
        # FIX: Passed unique key suffix
        plot_sales_over_time(historical_df, key_suffix="summary")

    with plot_col2:
        st.caption("Top Selling States")
        # FIX: Passed unique key suffix
        plot_sales_by_state(historical_df, top_n=5, key_suffix="summary")

# --- Main Application Logic ---

def main():
    """The main function to run the Streamlit app."""

    st.title("🚀 Executive Retail Sales & Forecasting Dashboard")
    st.markdown(
        """
        This modular dashboard separates the heavy **Prophet Model** training (backend) from the **visualization** (frontend) 
        for a faster, more reliable user experience.
        """
    )
    st.markdown("---")
    
    # 1. Load Data
    historical_data = load_historical_data(FILE_NAME)
    forecast_data = load_forecast_data(FORECAST_RESULTS_FILE)
    
    # 2. Calculate Metrics
    kpis = calculate_kpis(historical_data, forecast_data)

    # --- Sidebar for Filters and Settings ---
    st.sidebar.title("Configuration & Filters")
    
    # Filter by State
    selected_states = st.sidebar.multiselect(
        "Filter Data by State (Affects EDA Tab Only)",
        options=historical_data['State'].unique(),
        default=[] 
    )
    
    # Display forecast status
    if forecast_data is None:
        st.sidebar.error("FORECAST UNAVAILABLE: Please run `train_and_save_model.py`")
    else:
        st.sidebar.success(f"Forecast Loaded: {len(forecast_data)} periods")
        
    # Apply State Filter for EDA
    if selected_states:
        filtered_data = historical_data[historical_data['State'].isin(selected_states)].copy()
        st.sidebar.info(f"EDA filtered to {len(selected_states)} state(s).")
    else:
        filtered_data = historical_data.copy()
        st.sidebar.info("EDA showing all states.")

    # --- Main Content Tabs ---
    
    tab0, tab1, tab2 = st.tabs([
        "⭐ Executive Summary", 
        "📈 Detailed Data Visualizations (EDA)", 
        "🔮 Sales Forecasting Results"
    ])

    with tab0:
        executive_summary_tab(kpis, historical_data)

    with tab1:
        st.header("Detailed Exploratory Data Analysis (EDA)")
        
        # Display plots side-by-side in this tab
        col_hist, col_state = st.columns(2)
        with col_hist:
            # FIX: Passed unique key suffix
            plot_sales_over_time(filtered_data, key_suffix="detailed_time")
        with col_state:
            # FIX: Passed unique key suffix
            plot_sales_by_state(filtered_data, top_n=10, key_suffix="detailed_state")
        
        st.markdown("---")
        st.subheader("Raw Data Sample (Top 10 Rows)")
        st.dataframe(filtered_data.head(10))


    with tab2:
        st.header("Time Series Forecasting Results")
        
        if forecast_data is not None:
            # The forecast plot data function is only called once, so the key is handled internally
            plot_forecast_data(historical_data, forecast_data)
            st.markdown("---")
            st.info("The forecast was generated by running `train_and_save_model.py` separately.")
        else:
            st.warning("Please run the `train_and_save_model.py` script first to generate the necessary forecast data.")


if __name__ == "__main__":
    main()
