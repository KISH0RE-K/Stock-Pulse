import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Color Palette Definitions (Fintech Dark Theme)
BG_COLOR = "rgba(17, 24, 39, 1)"  # Tailwind gray-900
GRID_COLOR = "rgba(75, 85, 99, 0.2)" # Tailwind gray-600 with low opacity
TEXT_COLOR = "#f3f4f6"  # Tailwind gray-100
BLUE_PRIMARY = "#00b4d8"
TEAL_PRIMARY = "#00f5d4"
ACCENT_MUTED = "#90e0ef"

def get_dark_layout_template():
    """
    Returns a unified layout dictionary for dark-themed fintech plots.
    """
    return dict(
        paper_bgcolor="rgba(0,0,0,0)", # Transparent to match Streamlit dark background
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR, family="Inter, Roboto, sans-serif"),
        xaxis=dict(
            gridcolor=GRID_COLOR,
            linecolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            showgrid=True,
            tickfont=dict(color="#9ca3af")
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            linecolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            showgrid=True,
            tickfont=dict(color="#9ca3af")
        ),
        legend=dict(
            bgcolor="rgba(17, 24, 39, 0.8)",
            bordercolor=GRID_COLOR,
            font=dict(color=TEXT_COLOR)
        ),
        margin=dict(l=40, r=40, t=50, b=40)
    )

def plot_close_price(df):
    """
    Renders an interactive Line chart of AAPL Closing Price over time.
    """
    fig = go.Figure()
    
    # Clean up Date index/format
    dates = pd.to_datetime(df['Date'])
    
    # Area chart with gradient-like fill using scatter
    fig.add_trace(go.Scatter(
        x=dates,
        y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color=TEAL_PRIMARY, width=2.5),
        fill='tozeroy',
        fillcolor='rgba(0, 245, 212, 0.08)', # Semi-transparent teal
        hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Close</b>: $%{y:.2f}<extra></extra>'
    ))
    
    # Add a 20-day Moving Average line for professional look
    if 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=dates,
            y=df['SMA_20'],
            mode='lines',
            name='20-day SMA',
            line=dict(color=BLUE_PRIMARY, width=1.5, dash='dash'),
            hovertemplate='<b>20-day SMA</b>: $%{y:.2f}<extra></extra>'
        ))

    # Extract year dynamically
    data_year = pd.to_datetime(df['Date']).dt.year.iloc[0] if len(df) > 0 else ""
    fig.update_layout(
        title=dict(text=f"<b>Closing Price Over Time ({data_year})</b>", font=dict(size=16, color=TEXT_COLOR)),
        hovermode="x unified",
        **get_dark_layout_template()
    )
    return fig

def plot_monthly_avg(df):
    """
    Renders a Bar chart of monthly average closing price.
    """
    # Group by month chronologically
    df_temp = df.copy()
    df_temp['Date'] = pd.to_datetime(df_temp['Date'])
    df_temp['Month_Num'] = df_temp['Date'].dt.month
    df_temp['Month_Name'] = df_temp['Date'].dt.strftime('%b')
    
    monthly_avg = df_temp.groupby(['Month_Num', 'Month_Name'])['Close'].mean().reset_index()
    monthly_avg = monthly_avg.sort_values('Month_Num')
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly_avg['Month_Name'],
        y=monthly_avg['Close'],
        marker=dict(
            color=monthly_avg['Close'],
            colorscale=[[0.0, '#0077b6'], [1.0, TEAL_PRIMARY]],
            line=dict(color='rgba(17, 24, 39, 1)', width=1)
        ),
        hovertemplate='<b>Month</b>: %{x}<br><b>Avg Close</b>: $%{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text="<b>Monthly Average Closing Price</b>", font=dict(size=16, color=TEXT_COLOR)),
        **get_dark_layout_template()
    )
    return fig

def plot_returns_dist(df):
    """
    Renders a Histogram of Daily Returns distribution.
    """
    # Daily returns in percentage
    returns_pct = df['Daily_Return'].dropna() * 100
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns_pct,
        nbinsx=40,
        marker=dict(
            color=BLUE_PRIMARY,
            line=dict(color='rgba(17, 24, 39, 1)', width=0.5)
        ),
        opacity=0.75,
        name='Daily Return (%)',
        hovertemplate='<b>Return Range</b>: %{x:.2f}%<br><b>Frequency</b>: %{y}<extra></extra>'
    ))
    
    # Add vertical line at 0 (neutral mark)
    fig.add_vline(x=0.0, line_width=1.5, line_dash="dash", line_color=TEAL_PRIMARY)
    
    fig.update_layout(
        title=dict(text="<b>Daily Returns Distribution (%)</b>", font=dict(size=16, color=TEXT_COLOR)),
        xaxis_title="Daily Return (%)",
        yaxis_title="Count",
        **get_dark_layout_template()
    )
    return fig

def plot_feature_importance(importances_df):
    """
    Renders a Horizontal Bar Chart for Random Forest feature importances.
    """
    # Sort just in case
    df_sorted = importances_df.sort_values('Importance', ascending=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_sorted['Importance'],
        y=df_sorted['Feature'],
        orientation='h',
        marker=dict(
            color=df_sorted['Importance'],
            colorscale=[[0.0, '#023e8a'], [1.0, TEAL_PRIMARY]],
            line=dict(color='rgba(17, 24, 39, 1)', width=1)
        ),
        hovertemplate='<b>Feature</b>: %{y}<br><b>Importance</b>: %{x:.2%}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text="<b>Feature Importance (Random Forest Classifier)</b>", font=dict(size=16, color=TEXT_COLOR)),
        **get_dark_layout_template()
    )
    fig.update_xaxes(tickformat='.1%')
    return fig
