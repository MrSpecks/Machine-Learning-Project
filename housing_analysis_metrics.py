"""
Housing Analysis Metrics Module

This module contains business logic functions for calculating metrics,
statistics, and insights from the Ames Housing dataset. These functions
are designed to work with filtered datasets for dashboard applications.

Author: Kagiso Mfusi
Date: October 2024
Version: 2.0 (Refactored for modularity)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats


def get_top_correlations(df: pd.DataFrame, target_col: str = 'SalePrice', 
                        n_top: int = 10) -> pd.Series:
    """
    Get top correlations with the target variable.
    
    Args:
        df (pd.DataFrame): Housing dataset
        target_col (str): Target column name
        n_top (int): Number of top correlations to return
        
    Returns:
        pd.Series: Top correlations with target variable
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    # Get numerical columns only
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    numerical_cols = [col for col in numerical_cols if col != target_col]
    
    # Calculate correlations
    correlations = df[numerical_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
    
    return correlations.head(n_top)


def calculate_neighborhood_stats(df: pd.DataFrame, 
                              target_col: str = 'SalePrice') -> pd.DataFrame:
    """
    Calculate neighborhood statistics including count, mean, median, and std.
    
    Args:
        df (pd.DataFrame): Housing dataset
        target_col (str): Target column name
        
    Returns:
        pd.DataFrame: Neighborhood statistics
    """
    if 'Neighborhood' not in df.columns:
        raise ValueError("Neighborhood column not found in dataset")
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    stats_df = df.groupby('Neighborhood')[target_col].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(2)
    
    stats_df.columns = ['Count', 'Mean_Price', 'Median_Price', 'Std_Price', 'Min_Price', 'Max_Price']
    stats_df = stats_df.sort_values('Median_Price', ascending=False)
    
    return stats_df


def get_median_price_by_quality(df: pd.DataFrame, 
                               target_col: str = 'SalePrice') -> pd.DataFrame:
    """
    Get median prices by overall quality rating.
    
    Args:
        df (pd.DataFrame): Housing dataset
        target_col (str): Target column name
        
    Returns:
        pd.DataFrame: Quality-based price statistics
    """
    if 'OverallQual' not in df.columns:
        raise ValueError("OverallQual column not found in dataset")
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    quality_stats = df.groupby('OverallQual')[target_col].agg([
        'count', 'mean', 'median', 'std'
    ]).round(2)
    
    quality_stats.columns = ['Count', 'Mean_Price', 'Median_Price', 'Std_Price']
    
    return quality_stats


def get_price_distribution_stats(df: pd.DataFrame, 
                                target_col: str = 'SalePrice') -> Dict[str, float]:
    """
    Get comprehensive price distribution statistics.
    
    Args:
        df (pd.DataFrame): Housing dataset
        target_col (str): Target column name
        
    Returns:
        Dict[str, float]: Price distribution statistics
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    prices = df[target_col]
    
    stats_dict = {
        'count': len(prices),
        'mean': prices.mean(),
        'median': prices.median(),
        'std': prices.std(),
        'min': prices.min(),
        'max': prices.max(),
        'q25': prices.quantile(0.25),
        'q75': prices.quantile(0.75),
        'skewness': stats.skew(prices),
        'kurtosis': stats.kurtosis(prices),
        'cv': prices.std() / prices.mean()  # Coefficient of variation
    }
    
    # Round numerical values
    for key, value in stats_dict.items():
        if isinstance(value, (int, float)):
            stats_dict[key] = round(value, 2)
    
    return stats_dict


def detect_outliers_iqr(df: pd.DataFrame, columns: List[str], 
                       multiplier: float = 1.5) -> Dict[str, Dict]:
    """
    Detect outliers using the Interquartile Range (IQR) method.
    
    Args:
        df (pd.DataFrame): Housing dataset
        columns (List[str]): Columns to check for outliers
        multiplier (float): IQR multiplier for outlier detection
        
    Returns:
        Dict[str, Dict]: Outlier information for each column
    """
    outlier_info = {}
    
    for col in columns:
        if col not in df.columns:
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        outlier_info[col] = {
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_count': len(outliers),
            'outlier_percentage': len(outliers) / len(df) * 100,
            'outlier_indices': outliers.index.tolist()
        }
    
    return outlier_info


def get_feature_importance_scores(df: pd.DataFrame, 
                                 target_col: str = 'SalePrice',
                                 method: str = 'correlation') -> pd.Series:
    """
    Calculate feature importance scores using different methods.
    
    Args:
        df (pd.DataFrame): Housing dataset
        target_col (str): Target column name
        method (str): Method to use ('correlation', 'mutual_info')
        
    Returns:
        pd.Series: Feature importance scores
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    # Get numerical columns only
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    numerical_cols = [col for col in numerical_cols if col != target_col]
    
    if method == 'correlation':
        # Use absolute correlation as importance score
        importance_scores = df[numerical_cols].corrwith(df[target_col]).abs()
    elif method == 'mutual_info':
        # Use mutual information (requires sklearn)
        try:
            from sklearn.feature_selection import mutual_info_regression
            X = df[numerical_cols].fillna(0)
            y = df[target_col]
            scores = mutual_info_regression(X, y)
            importance_scores = pd.Series(scores, index=numerical_cols)
        except ImportError:
            print("Warning: sklearn not available, falling back to correlation method")
            importance_scores = df[numerical_cols].corrwith(df[target_col]).abs()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return importance_scores.sort_values(ascending=False)


def calculate_price_trends_by_year(df: pd.DataFrame, 
                                   target_col: str = 'SalePrice') -> pd.DataFrame:
    """
    Calculate price trends by year sold.
    
    Args:
        df (pd.DataFrame): Housing dataset
        target_col (str): Target column name
        
    Returns:
        pd.DataFrame: Year-based price statistics
    """
    if 'YrSold' not in df.columns:
        raise ValueError("YrSold column not found in dataset")
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    year_stats = df.groupby('YrSold')[target_col].agg([
        'count', 'mean', 'median', 'std'
    ]).round(2)
    
    year_stats.columns = ['Count', 'Mean_Price', 'Median_Price', 'Std_Price']
    
    return year_stats


def get_size_price_analysis(df: pd.DataFrame, 
                          size_col: str = 'TotalSF',
                          target_col: str = 'SalePrice') -> Dict[str, float]:
    """
    Analyze the relationship between size and price.
    
    Args:
        df (pd.DataFrame): Housing dataset
        size_col (str): Size column name
        target_col (str): Target column name
        
    Returns:
        Dict[str, float]: Size-price analysis metrics
    """
    if size_col not in df.columns:
        raise ValueError(f"Size column '{size_col}' not found in dataset")
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    # Calculate correlation
    correlation = df[size_col].corr(df[target_col])
    
    # Calculate price per square foot if both columns exist
    price_per_sf = None
    if size_col in df.columns and target_col in df.columns:
        price_per_sf = df[target_col].mean() / df[size_col].mean()
    
    # Size categories analysis
    size_quartiles = df[size_col].quantile([0.25, 0.5, 0.75])
    
    analysis = {
        'correlation': correlation,
        'avg_price_per_sf': price_per_sf,
        'size_q25': size_quartiles[0.25],
        'size_q50': size_quartiles[0.5],
        'size_q75': size_quartiles[0.75],
        'min_size': df[size_col].min(),
        'max_size': df[size_col].max(),
        'mean_size': df[size_col].mean()
    }
    
    # Round numerical values
    for key, value in analysis.items():
        if isinstance(value, (int, float)) and not pd.isna(value):
            analysis[key] = round(value, 2)
    
    return analysis


def get_quality_price_analysis(df: pd.DataFrame, 
                              quality_col: str = 'OverallQual',
                              target_col: str = 'SalePrice') -> pd.DataFrame:
    """
    Analyze the relationship between quality and price.
    
    Args:
        df (pd.DataFrame): Housing dataset
        quality_col (str): Quality column name
        target_col (str): Target column name
        
    Returns:
        pd.DataFrame: Quality-price analysis
    """
    if quality_col not in df.columns:
        raise ValueError(f"Quality column '{quality_col}' not found in dataset")
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    quality_analysis = df.groupby(quality_col)[target_col].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(2)
    
    quality_analysis.columns = ['Count', 'Mean_Price', 'Median_Price', 'Std_Price', 'Min_Price', 'Max_Price']
    
    return quality_analysis


def get_location_price_analysis(df: pd.DataFrame, 
                              location_col: str = 'Neighborhood',
                              target_col: str = 'SalePrice') -> pd.DataFrame:
    """
    Analyze price variations by location.
    
    Args:
        df (pd.DataFrame): Housing dataset
        location_col (str): Location column name
        target_col (str): Target column name
        
    Returns:
        pd.DataFrame: Location-based price analysis
    """
    if location_col not in df.columns:
        raise ValueError(f"Location column '{location_col}' not found in dataset")
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    location_analysis = df.groupby(location_col)[target_col].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(2)
    
    location_analysis.columns = ['Count', 'Mean_Price', 'Median_Price', 'Std_Price', 'Min_Price', 'Max_Price']
    location_analysis = location_analysis.sort_values('Median_Price', ascending=False)
    
    return location_analysis


def get_feature_correlation_matrix(df: pd.DataFrame, 
                                 features: List[str]) -> pd.DataFrame:
    """
    Get correlation matrix for specified features.
    
    Args:
        df (pd.DataFrame): Housing dataset
        features (List[str]): List of features to include
        
    Returns:
        pd.DataFrame: Correlation matrix
    """
    # Filter to existing features
    existing_features = [f for f in features if f in df.columns]
    
    if not existing_features:
        raise ValueError("None of the specified features found in dataset")
    
    # Get numerical features only
    numerical_features = df[existing_features].select_dtypes(include=[np.number]).columns.tolist()
    
    if not numerical_features:
        raise ValueError("No numerical features found in the specified list")
    
    correlation_matrix = df[numerical_features].corr()
    
    return correlation_matrix


def get_summary_statistics(df: pd.DataFrame, 
                          target_col: str = 'SalePrice') -> Dict[str, Union[str, int, float]]:
    """
    Get comprehensive summary statistics for the dataset.
    
    Args:
        df (pd.DataFrame): Housing dataset
        target_col (str): Target column name
        
    Returns:
        Dict[str, Union[str, int, float]]: Summary statistics
    """
    summary = {
        'total_records': len(df),
        'total_features': len(df.columns),
        'numerical_features': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_features': len(df.select_dtypes(include=['object']).columns),
        'missing_values': df.isnull().sum().sum(),
        'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    }
    
    if target_col in df.columns:
        price_stats = get_price_distribution_stats(df, target_col)
        summary.update({
            'mean_price': price_stats['mean'],
            'median_price': price_stats['median'],
            'price_range': price_stats['max'] - price_stats['min'],
            'price_std': price_stats['std'],
            'price_cv': price_stats['cv']
        })
    
    # Round numerical values
    for key, value in summary.items():
        if isinstance(value, (int, float)) and not pd.isna(value):
            summary[key] = round(value, 2)
    
    return summary


def filter_dataset(df: pd.DataFrame, 
                  filters: Dict[str, Union[str, List, Tuple]]) -> pd.DataFrame:
    """
    Filter dataset based on specified criteria.
    
    Args:
        df (pd.DataFrame): Housing dataset
        filters (Dict[str, Union[str, List, Tuple]]): Filter criteria
        
    Returns:
        pd.DataFrame: Filtered dataset
        
    Example:
        >>> filters = {
        ...     'Neighborhood': ['Northridge', 'Stone Brook'],
        ...     'OverallQual': (7, 10),  # Range
        ...     'YearBuilt': (2000, 2010)  # Range
        ... }
        >>> filtered_df = filter_dataset(df, filters)
    """
    filtered_df = df.copy()
    
    for column, criteria in filters.items():
        if column not in df.columns:
            print(f"Warning: Column '{column}' not found in dataset")
            continue
        
        if isinstance(criteria, (list, tuple)) and len(criteria) == 2:
            # Range filter (min, max)
            min_val, max_val = criteria
            filtered_df = filtered_df[(filtered_df[column] >= min_val) & 
                                   (filtered_df[column] <= max_val)]
        elif isinstance(criteria, list):
            # List filter (in values)
            filtered_df = filtered_df[filtered_df[column].isin(criteria)]
        else:
            # Single value filter
            filtered_df = filtered_df[filtered_df[column] == criteria]
    
    return filtered_df


if __name__ == "__main__":
    # Example usage
    from housing_data_loader import get_analysis_ready_data
    
    # Load data
    df = get_analysis_ready_data()
    
    # Example analyses
    print("📊 Top Correlations with SalePrice:")
    top_corr = get_top_correlations(df, n_top=5)
    print(top_corr)
    
    print("\n🏘️ Neighborhood Statistics:")
    neighborhood_stats = calculate_neighborhood_stats(df)
    print(neighborhood_stats.head())
    
    print("\n📈 Price Distribution Stats:")
    price_stats = get_price_distribution_stats(df)
    print(price_stats)
    
    print("\n📋 Dataset Summary:")
    summary = get_summary_statistics(df)
    for key, value in summary.items():
        print(f"   {key}: {value}")
