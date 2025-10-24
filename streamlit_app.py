"""
Ames Housing Market Dashboard

A professional Streamlit dashboard for analyzing the Ames Housing dataset.
Uses modular functions from housing_data_loader.py and housing_analysis_metrics.py
to provide interactive insights for home buyers, sellers, and real estate agents.

Author: Kagiso Mfusi
Date: October 2024
Version: 1.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Import our custom modules
from housing_data_loader import get_analysis_ready_data, get_feature_summary
from housing_analysis_metrics import (
    get_top_correlations, calculate_neighborhood_stats, get_median_price_by_quality,
    get_price_distribution_stats, filter_dataset, get_summary_statistics
)

# Page configuration
st.set_page_config(
    page_title="Ames Housing Market Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .trend-up {
        color: #28a745;
        font-weight: bold;
    }
    .trend-down {
        color: #dc3545;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the housing data."""
    return get_analysis_ready_data("house-prices-advanced-regression-techniques/train.csv")

@st.cache_data
def get_baseline_stats(df):
    """Get baseline statistics for comparison."""
    return {
        'median_price': df['SalePrice'].median(),
        'avg_price_per_sqft': (df['SalePrice'] / df['GrLivArea']).mean(),
        'total_properties': len(df),
        'avg_quality': df['OverallQual'].mean(),
        'avg_age': (df['YrSold'] - df['YearBuilt']).mean(),
        'avg_garage_cars': df['GarageCars'].mean()
    }

def format_currency(value):
    """Format currency values."""
    if value >= 1000000:
        return f"${value/1000000:.1f}M"
    elif value >= 1000:
        return f"${value/1000:.0f}K"
    else:
        return f"${value:.0f}"

def format_number(value, decimals=1):
    """Format numbers with appropriate precision."""
    return f"{value:.{decimals}f}"

def get_trend_indicator(current, baseline, reverse=False):
    """Get trend indicator (up/down arrow) based on comparison."""
    if reverse:
        is_better = current < baseline
    else:
        is_better = current > baseline
    
    if is_better:
        return "📈", "trend-up"
    else:
        return "📉", "trend-down"

def main():
    # Header
    st.markdown('<h1 class="main-header">🏠 Ames Housing Market Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading housing data..."):
        df = load_data()
        baseline_stats = get_baseline_stats(df)
    
    # Sidebar filters
    st.sidebar.header("🔍 Global Filters")
    
    # Neighborhood filter
    neighborhoods = sorted(df['Neighborhood'].unique())
    selected_neighborhoods = st.sidebar.multiselect(
        "Neighborhood",
        options=neighborhoods,
        default=neighborhoods[:5],  # Default to first 5 neighborhoods
        help="Select one or more neighborhoods to analyze"
    )
    
    # Overall Quality filter
    quality_range = st.sidebar.slider(
        "Overall Quality Range",
        min_value=1,
        max_value=10,
        value=(1, 10),
        help="Filter by overall quality rating (1-10)"
    )
    
    # Year Sold filter
    years_sold = sorted(df['YrSold'].unique())
    selected_years = st.sidebar.multiselect(
        "Year Sold",
        options=years_sold,
        default=years_sold,
        help="Select years when properties were sold"
    )
    
    # Price Range filter
    price_min = int(df['SalePrice'].min())
    price_max = int(df['SalePrice'].max())
    price_range = st.sidebar.slider(
        "Price Range",
        min_value=price_min,
        max_value=price_max,
        value=(price_min, price_max),
        format="$%d",
        help="Filter by sale price range"
    )
    
    # Apply filters
    filters = {}
    if selected_neighborhoods:
        filters['Neighborhood'] = selected_neighborhoods
    if quality_range != (1, 10):
        filters['OverallQual'] = quality_range
    if selected_years:
        filters['YrSold'] = selected_years
    if price_range != (price_min, price_max):
        filters['SalePrice'] = price_range
    
    # Filter the dataset
    if filters:
        filtered_df = filter_dataset(df, filters)
    else:
        filtered_df = df.copy()
    
    # Check if filtered data is empty
    if len(filtered_df) == 0:
        st.error("No data matches the selected filters. Please adjust your filter criteria.")
        return
    
    # Calculate filtered statistics
    filtered_stats = {
        'median_price': filtered_df['SalePrice'].median(),
        'avg_price_per_sqft': (filtered_df['SalePrice'] / filtered_df['GrLivArea']).mean(),
        'total_properties': len(filtered_df),
        'avg_quality': filtered_df['OverallQual'].mean(),
        'avg_age': (filtered_df['YrSold'] - filtered_df['YearBuilt']).mean(),
        'avg_garage_cars': filtered_df['GarageCars'].mean()
    }
    
    # KPI Row
    st.subheader("📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Median Sale Price
        median_price = filtered_stats['median_price']
        baseline_median = baseline_stats['median_price']
        trend_icon, trend_class = get_trend_indicator(median_price, baseline_median)
        
        st.metric(
            label="Median Sale Price",
            value=format_currency(median_price),
            delta=f"{trend_icon} {format_currency(abs(median_price - baseline_median))} vs overall"
        )
    
    with col2:
        # Average Price per SqFt
        avg_price_per_sqft = filtered_stats['avg_price_per_sqft']
        baseline_price_per_sqft = baseline_stats['avg_price_per_sqft']
        trend_icon, trend_class = get_trend_indicator(avg_price_per_sqft, baseline_price_per_sqft)
        
        st.metric(
            label="Avg. Price per SqFt",
            value=f"${avg_price_per_sqft:.0f}",
            delta=f"{trend_icon} ${abs(avg_price_per_sqft - baseline_price_per_sqft):.0f} vs overall"
        )
    
    with col3:
        # Total Properties Sold
        total_properties = filtered_stats['total_properties']
        baseline_total = baseline_stats['total_properties']
        trend_icon, trend_class = get_trend_indicator(total_properties, baseline_total)
        
        st.metric(
            label="Total Properties Sold",
            value=f"{total_properties:,}",
            delta=f"{trend_icon} {abs(total_properties - baseline_total):,} vs overall"
        )
    
    with col4:
        # Average Overall Quality
        avg_quality = filtered_stats['avg_quality']
        baseline_quality = baseline_stats['avg_quality']
        trend_icon, trend_class = get_trend_indicator(avg_quality, baseline_quality)
        
        st.metric(
            label="Average Overall Quality",
            value=f"{avg_quality:.1f}/10",
            delta=f"{trend_icon} {abs(avg_quality - baseline_quality):.1f} vs overall"
        )
    
    # Charts Grid (2x2 Layout)
    st.subheader("📈 Market Analysis Charts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price vs. Size Scatter Plot
        fig_scatter = px.scatter(
            filtered_df,
            x='GrLivArea',
            y='SalePrice',
            color='OverallQual',
            color_continuous_scale='Viridis',
            title="Price vs. Living Area (Colored by Quality)",
            labels={
                'GrLivArea': 'Above-Grade Living Area (SqFt)',
                'SalePrice': 'Sale Price ($)',
                'OverallQual': 'Overall Quality'
            },
            hover_data=['Neighborhood', 'YearBuilt', 'GarageCars']
        )
        
        # Add trend line
        z = np.polyfit(filtered_df['GrLivArea'], filtered_df['SalePrice'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(filtered_df['GrLivArea'].min(), filtered_df['GrLivArea'].max(), 100)
        fig_scatter.add_trace(go.Scatter(
            x=x_trend,
            y=p(x_trend),
            mode='lines',
            name='Trend Line',
            line=dict(color='red', dash='dash')
        ))
        
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Sale Price Distribution by Neighborhood (Box Plot)
        if selected_neighborhoods:
            neighborhood_data = filtered_df[filtered_df['Neighborhood'].isin(selected_neighborhoods)]
            
            # Calculate median prices for sorting
            neighborhood_medians = neighborhood_data.groupby('Neighborhood')['SalePrice'].median().sort_values(ascending=False)
            
            fig_box = px.box(
                neighborhood_data,
                x='Neighborhood',
                y='SalePrice',
                title="Sale Price Distribution by Neighborhood",
                labels={
                    'SalePrice': 'Sale Price ($)',
                    'Neighborhood': 'Neighborhood'
                }
            )
            
            # Sort by median price
            fig_box.update_xaxes(categoryorder='array', categoryarray=neighborhood_medians.index)
            fig_box.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("Please select neighborhoods to view the distribution chart.")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Median Price by Quality (Bar Chart)
        quality_stats = get_median_price_by_quality(filtered_df)
        
        fig_bar = px.bar(
            x=quality_stats.index,
            y=quality_stats['Median_Price'],
            title="Median Sale Price by Overall Quality",
            labels={
                'x': 'Overall Quality Rating',
                'y': 'Median Sale Price ($)'
            }
        )
        
        fig_bar.update_layout(height=400)
        fig_bar.update_yaxes(tickformat='$,.0f')
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col4:
        # Key Feature Correlation Heatmap
        top_features = ['SalePrice', 'OverallQual', 'TotalSF', 'GrLivArea', 'GarageCars', 
                       'GarageArea', 'TotalBsmtSF', 'TotalBath', 'HouseAge', 'YearBuilt',
                       'QualitySize', 'QualityLivArea', 'BathBedRatio']
        
        # Filter to existing features
        existing_features = [f for f in top_features if f in filtered_df.columns]
        
        if len(existing_features) > 1:
            corr_matrix = filtered_df[existing_features].corr()
            
            fig_heatmap = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Key Feature Correlation with SalePrice",
                color_continuous_scale='RdBu',
                color_continuous_midpoint=0
            )
            
            fig_heatmap.update_layout(height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("Insufficient features for correlation analysis.")
    
    # Bottom Row: Additional Stats
    st.subheader("📋 Additional Statistics")
    
    col5, col6 = st.columns(2)
    
    with col5:
        # Average Age of Property
        avg_age = filtered_stats['avg_age']
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="margin: 0; font-size: 2rem; color: #1f77b4;">{format_number(avg_age)} Years</h2>
            <p style="margin: 0.5rem 0 0 0; color: #666;">Average Property Age at Sale</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        # Average Garage Capacity
        avg_garage_cars = filtered_stats['avg_garage_cars']
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="margin: 0; font-size: 2rem; color: #1f77b4;">{format_number(avg_garage_cars)} Cars</h2>
            <p style="margin: 0.5rem 0 0 0; color: #666;">Average Garage Capacity</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>🏠 Ames Housing Market Dashboard | Built with Streamlit | Data: Kaggle House Prices Competition</p>
        <p>Author: Kagiso Mfusi | October 2024</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
