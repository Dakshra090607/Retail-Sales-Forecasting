Retail Sales Forecasting and Business Intelligence Platform

Project Overview

This project delivers a decoupled predictive analytics platform for retail sales forecasting and operational performance monitoring. It leverages the Prophet time series model for reliable 365-day sales predictions and a Streamlit dashboard for interactive visualization of historical KPIs and future forecasts.

The architecture separates the computationally intensive model training (backend) from the low-latency dashboard visualization (frontend) for optimal performance and user experience.

🚀 Key Features

Decoupled Architecture: Model training runs offline and saves results to a CSV, allowing the Streamlit dashboard to load instantly.

Prophet Forecasting: Utilizes the robust Prophet model to account for complex annual and weekly seasonality.

Interactive BI Dashboard: Provides key performance indicators (KPIs), detailed exploratory data analysis (EDA) with state-level filtering, and a visual forecast with uncertainty bounds.

Probabilistic Reporting: Forecast output includes a point prediction ($\hat{y}$) and upper/lower uncertainty bounds ($\hat{y}_{\text{upper}}$, $\hat{y}_{\text{lower}}$) for effective risk management.

🛠️ Technology Stack

Component

Technology

Rationale

Forecasting

Python, Prophet

Robust time series modeling for seasonality and trend decomposition.

Data Handling

Python, Pandas

Data preparation, aggregation, and I/O.

Visualization/UI

Python, Streamlit, Plotly

Fast deployment of a dynamic, interactive web dashboard.

Data Flow

CSV (sales\_forecast\_results.csv)

Simple, persistent storage for transferring forecast results.

📁 Project Structure

.
├── stores_sales_forecasting.csv  # 📥 Input Data: Raw historical sales transactions
├── train_and_save_model.py     # ⚙️ Modeling Script: Trains Prophet model and persists forecast
├── sales_dashboard.py          # 🖥️ Dashboard Script: Streamlit application for visualization
└── sales_forecast_results.csv  # 📊 Output Data: Generated forecast consumed by the dashboard


⚙️ Installation and Setup

Prerequisites

Ensure you have Python 3.8+ installed.

Step 1: Install Dependencies

This project requires Prophet, Streamlit, Pandas, and Plotly.

pip install prophet streamlit pandas plotly


Step 2: Prepare the Data

Ensure your historical sales data is named stores_sales_forecasting.csv and placed in the root directory.

🏃 Execution (Two-Step Process)

To run the application, the model must be trained first to generate the necessary forecast file.

Step 1: Generate the Sales Forecast (Backend)

Run the modeling script to process the data and save the 365-day forecast to sales_forecast_results.csv.

python train_and_save_model.py


Step 2: Launch the Business Intelligence Dashboard (Frontend)

Once the CSV file is created, start the Streamlit application to view the interactive dashboard.

streamlit run sales_dashboard.py


The dashboard will open automatically in your browser, providing access to the Executive Summary, Detailed EDA, and the Sales Forecasting Results tab.
