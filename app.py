import os
import sqlite3
import pandas as pd
import streamlit as st
import numpy as np

# Set Streamlit page config
st.set_page_config(
    page_title="AAPL Stock ETL & ML Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Dark Theme Styling
st.markdown("""
<style>
    /* Main Background and Text Colors */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Headers Styling */
    h1, h2, h3, h4, h5, h6 {
        color: #f1f5f9 !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }
    
    /* Styled HR dividers */
    .section-divider {
        border: 0;
        height: 1px;
        background: linear-gradient(to right, rgba(0, 180, 216, 0.7), rgba(0, 245, 212, 0.7), rgba(0, 180, 216, 0.05));
        margin: 2rem 0;
    }
    
    /* Card Container */
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 15px -3px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #00b4d8;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 0.5rem;
    }
    
    /* Custom Alert Banners */
    .banner-up {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.05));
        border: 1px solid #10b981;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.15);
        border-radius: 10px;
        padding: 1.5rem;
        color: #f8fafc;
    }
    .banner-down {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(220, 38, 38, 0.05));
        border: 1px solid #ef4444;
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.15);
        border-radius: 10px;
        padding: 1.5rem;
        color: #f8fafc;
    }
    
    /* Sidebar Styling */
    .css-164yf3f {
        background-color: #1e293b;
    }
</style>
""", unsafe_allow_html=True)

# Imports from project modules (delaying until after sys path setup if needed)
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from etl.extract import extract_stock_data
from etl.load import run_etl
from model.predict import train_and_predict
from visuals.charts import plot_close_price, plot_monthly_avg, plot_returns_dist, plot_feature_importance

DB_PATH = "database/stocks.db"
RAW_PATH = "data/raw/aapl_raw.csv"

# --- Sidebar Actions & Setup ---
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h2 style="color: #00f5d4 !important; margin-bottom: 0.2rem;">📈 AAPL Analytics</h2>
    <small style="color: #94a3b8; font-size: 0.85rem;">ETL & ML Prediction Engine</small>
</div>
""", unsafe_allow_html=True)

st.sidebar.subheader("ETL Control Center")

def trigger_full_etl():
    """Triggers yfinance extraction, clean, transform, and database insertion."""
    with st.sidebar.status("Running ETL Pipeline...", expanded=True) as status:
        st.write("1. Downloading raw AAPL stock data via yfinance...")
        extract_stock_data()
        st.write("2. Transforming and loading clean data into stocks.db...")
        run_etl()
        status.update(label="ETL Pipeline Complete!", state="complete", expanded=False)
    # Clear streamlit cache
    st.cache_data.clear()

if st.sidebar.button("⚡ Reload Data & Run ETL"):
    trigger_full_etl()

# Automate ETL run if the database doesn't exist yet
if not os.path.exists(RAW_PATH) or not os.path.exists(DB_PATH):
    trigger_full_etl()

# Display DB Stats in sidebar
if os.path.exists(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    db_size = os.path.getsize(DB_PATH) / 1024  # in KB
    rows_cnt = pd.read_sql_query("SELECT COUNT(*) as cnt FROM stocks", conn)['cnt'].values[0]
    conn.close()
    
    st.sidebar.markdown(f"""
    <hr style="border: 0; border-top: 1px solid #334155; margin: 1rem 0;"/>
    <h4 style="color: #00b4d8 !important; font-size: 1rem;">Database Status</h4>
    <div style="font-size: 0.85rem; color: #94a3b8;">
        <p style="margin: 0.2rem 0;">📁 File: <code>database/stocks.db</code></p>
        <p style="margin: 0.2rem 0;">📊 Row Count: <b>{rows_cnt}</b></p>
        <p style="margin: 0.2rem 0;">💾 Size: <b>{db_size:.1f} KB</b></p>
    </div>
    """, unsafe_allow_html=True)

# Main Title Header
st.markdown("""
<div style="margin-bottom: 2rem;">
    <h1 style="font-size: 2.5rem; margin-bottom: 0.2rem; background: linear-gradient(45deg, #00b4d8, #00f5d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        AAPL Stock Price ETL & ML Dashboard
    </h1>
    <p style="color: #94a3b8; font-size: 1.1rem; margin-top: 0;">Interactive financial analysis platform using Python, SQLite, Plotly, and Random Forest Classifier.</p>
</div>
""", unsafe_allow_html=True)

# Helper function to get clean dataframe
@st.cache_data
def get_data_from_db():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM stocks ORDER BY Date ASC", conn)
    conn.close()
    return df

try:
    df = get_data_from_db()
except Exception as e:
    st.error(f"Error loading database. Please click the ETL Reload button in the sidebar to reinitialize data. Details: {e}")
    st.stop()

# ==================== SECTION 1: ETL SUMMARY ====================
st.subheader("Section 1 - ETL Summary")

# Calculations
total_rows = len(df)
avg_close = df['Close'].mean()
high_price = df['High'].max()
low_price = df['Low'].min()
data_year = pd.to_datetime(df['Date']).dt.year.iloc[0] if len(df) > 0 else 2025

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Trading Days ({data_year})</div>
        <div class="metric-value" style="color: #00b4d8;">{total_rows}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Average Close Price</div>
        <div class="metric-value">${avg_close:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Highest Trading Price</div>
        <div class="metric-value" style="color: #10b981;">${high_price:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Lowest Trading Price</div>
        <div class="metric-value" style="color: #ef4444;">${low_price:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)

# ==================== SECTION 2: CHARTS ====================
st.subheader("Section 2 - Charts")

# Line chart spanning the full width
fig_line = plot_close_price(df)
st.plotly_chart(fig_line, use_container_width=True)

# Columns for side-by-side charts
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    fig_bar = plot_monthly_avg(df)
    st.plotly_chart(fig_bar, use_container_width=True)

with col_chart2:
    fig_hist = plot_returns_dist(df)
    st.plotly_chart(fig_hist, use_container_width=True)

st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)

# ==================== SECTION 3: SQL QUERIES ====================
st.subheader("Section 3 - SQL Queries")
st.markdown("<p style='color: #94a3b8;'>Running direct SQL queries against SQLite <code>database/stocks.db</code></p>", unsafe_allow_html=True)

# Executing SQL Queries
conn = sqlite3.connect(DB_PATH)

# Query 1: Average close price per month
q_avg_close_month = """
SELECT 
    strftime('%m', Date) as "Month Number",
    CASE strftime('%m', Date)
        WHEN '01' THEN 'January' WHEN '02' THEN 'February' WHEN '03' THEN 'March'
        WHEN '04' THEN 'April'   WHEN '05' THEN 'May'      WHEN '06' THEN 'June'
        WHEN '07' THEN 'July'    WHEN '08' THEN 'August'   WHEN '09' THEN 'September'
        WHEN '10' THEN 'October' WHEN '11' THEN 'November' WHEN '12' THEN 'December'
    END as "Month Name",
    ROUND(AVG(Close), 2) as "Avg Close Price ($)"
FROM stocks
GROUP BY "Month Number"
ORDER BY "Month Number"
"""
df_query1 = pd.read_sql_query(q_avg_close_month, conn)

# Query 2: Top 5 biggest price swing days
q_price_swings = """
SELECT 
    Date,
    ROUND(Open, 2) as "Open ($)",
    ROUND(High, 2) as "High ($)",
    ROUND(Low, 2) as "Low ($)",
    ROUND(Close, 2) as "Close ($)",
    ROUND(Price_Swing, 2) as "Swing ($)"
FROM stocks
ORDER BY Price_Swing DESC
LIMIT 5
"""
df_query2 = pd.read_sql_query(q_price_swings, conn)

# Query 3: Up days vs down days count
q_up_down_count = """
SELECT 
    CASE WHEN Close > Open THEN 'Up Day (Close > Open)' ELSE 'Down Day (Close <= Open)' END as "Day Outcome",
    COUNT(*) as "Count",
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM stocks), 1) || '%' as "Percentage"
FROM stocks
GROUP BY "Day Outcome"
"""
df_query3 = pd.read_sql_query(q_up_down_count, conn)

conn.close()

col_sql1, col_sql2, col_sql3 = st.columns(3)

with col_sql1:
    st.markdown("##### 📅 Monthly Average Close Price")
    st.dataframe(df_query1, use_container_width=True, hide_index=True)

with col_sql2:
    st.markdown("##### 🚀 Top 5 Price Swing Days")
    st.dataframe(df_query2, use_container_width=True, hide_index=True)

with col_sql3:
    st.markdown("##### ⚖️ Up Days vs Down Days")
    st.dataframe(df_query3, use_container_width=True, hide_index=True)

st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)

# ==================== SECTION 4: ML PREDICTION ====================
st.subheader("Section 4 - ML Prediction")

# Run ML Pipeline
with st.spinner("Training Random Forest Classifier model..."):
    try:
        ml_results = train_and_predict(DB_PATH)
        accuracy = ml_results['accuracy']
        prediction = ml_results['prediction']
        confidence = ml_results['confidence']
        conf_df = ml_results['confusion_matrix']
        feat_imp_df = ml_results['feature_importance']
        last_date = ml_results['last_date']
    except Exception as e:
        st.error(f"Error training Machine Learning model: {e}")
        st.stop()

col_ml1, col_ml2 = st.columns([1, 1.2])

with col_ml1:
    # Model Accuracy Card
    st.markdown(f"""
    <div class="metric-card" style="margin-bottom: 1.5rem;">
        <div class="metric-label">Model Accuracy (Test Set)</div>
        <div class="metric-value" style="color: #accbee; font-size: 2.2rem;">{accuracy:.2%}</div>
        <small style="color: #94a3b8;">Random Forest Classifier (sequential time series split)</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Prediction Alert Banner
    if prediction == "UP":
        st.markdown(f"""
        <div class="banner-up">
            <h3 style="color: #34d399 !important; margin: 0; font-size: 1.25rem;">📈 Bullish Outlook: Tomorrow will go UP</h3>
            <p style="margin: 0.5rem 0 0 0; color: #f8fafc; font-size: 1rem;">
                The model predicts the stock price will rise tomorrow (next trading day after <b>{last_date}</b>) 
                with a confidence of <b>{confidence:.1f}%</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="banner-down">
            <h3 style="color: #f87171 !important; margin: 0; font-size: 1.25rem;">📉 Bearish Outlook: Tomorrow will go DOWN</h3>
            <p style="margin: 0.5rem 0 0 0; color: #f8fafc; font-size: 1rem;">
                The model predicts the stock price will drop tomorrow (next trading day after <b>{last_date}</b>) 
                with a confidence of <b>{confidence:.1f}%</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Confusion Matrix display
    st.markdown("##### 🎯 Confusion Matrix (Model Predictions vs Actual)")
    st.table(conf_df)

with col_ml2:
    # Feature Importance plot
    fig_feat = plot_feature_importance(feat_imp_df)
    st.plotly_chart(fig_feat, use_container_width=True)

st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)

# ==================== SECTION 5: RAW DATA ====================
st.subheader("Section 5 - Raw Data")
st.markdown("<p style='color: #94a3b8;'>Cleaned and transformed stock data from SQLite database.</p>", unsafe_allow_html=True)

# Display interactive pandas dataframe
st.dataframe(
    df,
    use_container_width=True,
    column_config={
        "Date": st.column_config.TextColumn("Trading Date"),
        "Open": st.column_config.NumberColumn("Open ($)", format="$%.2f"),
        "High": st.column_config.NumberColumn("High ($)", format="$%.2f"),
        "Low": st.column_config.NumberColumn("Low ($)", format="$%.2f"),
        "Close": st.column_config.NumberColumn("Close ($)", format="$%.2f"),
        "Adj Close": st.column_config.NumberColumn("Adj Close ($)", format="$%.2f"),
        "Volume": st.column_config.NumberColumn("Volume", format="%d"),
        "Daily_Return": st.column_config.NumberColumn("Daily Return", format="%.4f"),
        "Price_Swing": st.column_config.NumberColumn("Price Swing ($)", format="$%.2f"),
    }
)
