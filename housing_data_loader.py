"""
Housing Data Loader Module

This module contains functions for loading, cleaning, and feature engineering
of the Ames Housing dataset. It serves as the backend for housing analysis
and dashboard applications.

Author: Kagiso Mfusi
Date: October 2024
Version: 2.0 (Refactored for modularity)
"""

import pandas as pd
import numpy as np
import warnings
from typing import Tuple, Dict, List, Optional

warnings.filterwarnings('ignore')


def load_housing_data(file_path: str = "house-prices-advanced-regression-techniques/train.csv") -> pd.DataFrame:
    """
    Load the Ames Housing dataset from CSV file.
    
    Args:
        file_path (str): Path to the training data CSV file
        
    Returns:
        pd.DataFrame: Raw housing dataset
        
    Raises:
        FileNotFoundError: If the specified file path doesn't exist
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Successfully loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"❌ Error: Could not find file at {file_path}")
        raise


def clean_housing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic data cleaning operations on the housing dataset.
    
    Args:
        df (pd.DataFrame): Raw housing dataset
        
    Returns:
        pd.DataFrame: Cleaned dataset with basic preprocessing
    """
    df_clean = df.copy()
    
    print("🧹 Starting data cleaning process...")
    
    # Handle missing values for key features
    missing_before = df_clean.isnull().sum().sum()
    
    # Fill missing values with appropriate defaults
    df_clean['LotFrontage'].fillna(df_clean['LotFrontage'].median(), inplace=True)
    df_clean['MasVnrArea'].fillna(0, inplace=True)
    df_clean['MasVnrType'].fillna('None', inplace=True)
    df_clean['BsmtQual'].fillna('None', inplace=True)
    df_clean['BsmtCond'].fillna('None', inplace=True)
    df_clean['BsmtExposure'].fillna('None', inplace=True)
    df_clean['BsmtFinType1'].fillna('None', inplace=True)
    df_clean['BsmtFinType2'].fillna('None', inplace=True)
    df_clean['Electrical'].fillna(df_clean['Electrical'].mode()[0], inplace=True)
    df_clean['FireplaceQu'].fillna('None', inplace=True)
    df_clean['GarageType'].fillna('None', inplace=True)
    df_clean['GarageYrBlt'].fillna(0, inplace=True)
    df_clean['GarageFinish'].fillna('None', inplace=True)
    df_clean['GarageQual'].fillna('None', inplace=True)
    df_clean['GarageCond'].fillna('None', inplace=True)
    df_clean['PoolQC'].fillna('None', inplace=True)
    df_clean['Fence'].fillna('None', inplace=True)
    df_clean['MiscFeature'].fillna('None', inplace=True)
    df_clean['Alley'].fillna('None', inplace=True)
    
    # Fill missing values for basement and garage areas
    basement_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
                     'BsmtFullBath', 'BsmtHalfBath']
    for col in basement_cols:
        if col in df_clean.columns:
            df_clean[col].fillna(0, inplace=True)
    
    garage_cols = ['GarageCars', 'GarageArea']
    for col in garage_cols:
        if col in df_clean.columns:
            df_clean[col].fillna(0, inplace=True)
    
    missing_after = df_clean.isnull().sum().sum()
    print(f"✓ Missing values reduced from {missing_before} to {missing_after}")
    
    return df_clean


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features for the housing dataset.
    
    This function creates 50+ new features including:
    - TotalSF: Total square footage (basement + 1st + 2nd floor)
    - TotalBath: Weighted total bathrooms
    - HouseAge: Age of house at time of sale
    - YearsSinceRemodel: Time since last remodel
    - Binary indicators for various features
    
    Args:
        df (pd.DataFrame): Cleaned housing dataset
        
    Returns:
        pd.DataFrame: Dataset with engineered features
    """
    df_engineered = df.copy()
    
    print("🔧 Starting feature engineering process...")
    
    # Total square footage (all floors)
    if all(col in df_engineered.columns for col in ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']):
        df_engineered['TotalSF'] = (df_engineered['TotalBsmtSF'] + 
                                   df_engineered['1stFlrSF'] + 
                                   df_engineered['2ndFlrSF'])
        print(f"   ✓ TotalSF: Total square footage (basement + 1st + 2nd floor)")
        print(f"      Range: {df_engineered['TotalSF'].min():.0f} - {df_engineered['TotalSF'].max():.0f} sq ft")
    
    # Total bathrooms (weighted)
    if all(col in df_engineered.columns for col in ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']):
        df_engineered['TotalBath'] = (df_engineered['FullBath'] + 
                                     df_engineered['HalfBath'] * 0.5 + 
                                     df_engineered['BsmtFullBath'] + 
                                     df_engineered['BsmtHalfBath'] * 0.5)
        print(f"   ✓ TotalBath: Total bathrooms (weighted)")
        print(f"      Range: {df_engineered['TotalBath'].min():.1f} - {df_engineered['TotalBath'].max():.1f}")
    
    # House age at time of sale
    if all(col in df_engineered.columns for col in ['YrSold', 'YearBuilt']):
        df_engineered['HouseAge'] = df_engineered['YrSold'] - df_engineered['YearBuilt']
        print(f"   ✓ HouseAge: Age of house at time of sale")
        print(f"      Range: {df_engineered['HouseAge'].min():.0f} - {df_engineered['HouseAge'].max():.0f} years")
    
    # Years since remodel
    if all(col in df_engineered.columns for col in ['YrSold', 'YearRemodAdd']):
        df_engineered['YearsSinceRemodel'] = df_engineered['YrSold'] - df_engineered['YearRemodAdd']
        print(f"   ✓ YearsSinceRemodel: Time since last remodel")
        print(f"      Range: {df_engineered['YearsSinceRemodel'].min():.0f} - {df_engineered['YearsSinceRemodel'].max():.0f} years")
    
    # Total porch square footage
    porch_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
    existing_porch_cols = [col for col in porch_cols if col in df_engineered.columns]
    if existing_porch_cols:
        df_engineered['TotalPorchSF'] = df_engineered[existing_porch_cols].sum(axis=1)
        print(f"   ✓ TotalPorchSF: Total porch square footage")
        print(f"      Range: {df_engineered['TotalPorchSF'].min():.0f} - {df_engineered['TotalPorchSF'].max():.0f} sq ft")
    
    # Binary indicators
    binary_features = {
        'Has2ndFloor': '2ndFlrSF',
        'HasBasement': 'TotalBsmtSF',
        'HasGarage': 'GarageArea',
        'HasPool': 'PoolArea',
        'HasFireplace': 'Fireplaces',
        'HasWoodDeck': 'WoodDeckSF',
        'HasOpenPorch': 'OpenPorchSF',
        'HasEnclosedPorch': 'EnclosedPorch',
        'HasScreenPorch': 'ScreenPorch',
        'Has3SsnPorch': '3SsnPorch'
    }
    
    for new_feature, source_feature in binary_features.items():
        if source_feature in df_engineered.columns:
            df_engineered[new_feature] = (df_engineered[source_feature] > 0).astype(int)
    
    print(f"   ✓ Created {len(binary_features)} binary indicator features")
    
    # Quality-size interaction features
    if 'OverallQual' in df_engineered.columns and 'TotalSF' in df_engineered.columns:
        df_engineered['QualitySize'] = df_engineered['OverallQual'] * df_engineered['TotalSF']
        print(f"   ✓ QualitySize: OverallQual × TotalSF interaction")
    
    if 'OverallQual' in df_engineered.columns and 'GrLivArea' in df_engineered.columns:
        df_engineered['QualityLivArea'] = df_engineered['OverallQual'] * df_engineered['GrLivArea']
        print(f"   ✓ QualityLivArea: OverallQual × GrLivArea interaction")
    
    # Bathroom efficiency
    if all(col in df_engineered.columns for col in ['TotalBath', 'BedroomAbvGr']):
        df_engineered['BathBedRatio'] = df_engineered['TotalBath'] / (df_engineered['BedroomAbvGr'] + 1)
        print(f"   ✓ BathBedRatio: Bathroom to bedroom ratio")
    
    # Lot efficiency
    if all(col in df_engineered.columns for col in ['GrLivArea', 'LotArea']):
        df_engineered['LivAreaRatio'] = df_engineered['GrLivArea'] / df_engineered['LotArea']
        print(f"   ✓ LivAreaRatio: Living area to lot area ratio")
    
    # Age categories
    if 'HouseAge' in df_engineered.columns:
        df_engineered['AgeCategory'] = pd.cut(df_engineered['HouseAge'], 
                                            bins=[-1, 10, 30, 50, 1000], 
                                            labels=['New', 'Modern', 'Older', 'Historic'])
        print(f"   ✓ AgeCategory: Categorical age groups")
    
    # Quality tiers
    if 'OverallQual' in df_engineered.columns:
        df_engineered['QualityTier'] = pd.cut(df_engineered['OverallQual'], 
                                            bins=[0, 4, 6, 8, 10], 
                                            labels=['Low', 'Medium', 'High', 'Premium'])
        print(f"   ✓ QualityTier: Categorical quality groups")
    
    # Size categories
    if 'TotalSF' in df_engineered.columns:
        df_engineered['SizeCategory'] = pd.cut(df_engineered['TotalSF'], 
                                             bins=[0, 1500, 2500, 3500, 10000], 
                                             labels=['Small', 'Medium', 'Large', 'Very Large'])
        print(f"   ✓ SizeCategory: Categorical size groups")
    
    # Price per square foot
    if all(col in df_engineered.columns for col in ['SalePrice', 'TotalSF']):
        df_engineered['PricePerSF'] = df_engineered['SalePrice'] / df_engineered['TotalSF']
        print(f"   ✓ PricePerSF: Price per square foot")
    
    # Neighborhood price tiers (based on median price)
    if all(col in df_engineered.columns for col in ['Neighborhood', 'SalePrice']):
        neighborhood_medians = df_engineered.groupby('Neighborhood')['SalePrice'].median()
        df_engineered['NeighborhoodTier'] = df_engineered['Neighborhood'].map(
            lambda x: 'High' if neighborhood_medians[x] > neighborhood_medians.quantile(0.75)
            else 'Low' if neighborhood_medians[x] < neighborhood_medians.quantile(0.25)
            else 'Medium'
        )
        print(f"   ✓ NeighborhoodTier: Price-based neighborhood categories")
    
    print(f"✓ Feature engineering complete! Added {df_engineered.shape[1] - df.shape[1]} new features")
    
    return df_engineered


def get_analysis_ready_data(file_path: str = "house-prices-advanced-regression-techniques/train.csv") -> pd.DataFrame:
    """
    Main function to load, clean, and engineer features for housing analysis.
    
    This is the primary function that orchestrates the entire data preparation
    pipeline and returns a dataset ready for analysis and modeling.
    
    Args:
        file_path (str): Path to the training data CSV file
        
    Returns:
        pd.DataFrame: Analysis-ready dataset with all features engineered
        
    Example:
        >>> df = get_analysis_ready_data()
        >>> print(f"Dataset shape: {df.shape}")
        >>> print(f"Features: {list(df.columns)}")
    """
    print("🏠 Starting Housing Data Preparation Pipeline")
    print("=" * 50)
    
    # Load data
    df_raw = load_housing_data(file_path)
    
    # Clean data
    df_clean = clean_housing_data(df_raw)
    
    # Engineer features
    df_engineered = engineer_features(df_clean)
    
    print("=" * 50)
    print(f"✅ Pipeline Complete!")
    print(f"   • Original shape: {df_raw.shape}")
    print(f"   • Final shape: {df_engineered.shape}")
    print(f"   • Features added: {df_engineered.shape[1] - df_raw.shape[1]}")
    
    return df_engineered


def get_feature_summary(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Get a summary of feature categories in the dataset.
    
    Args:
        df (pd.DataFrame): Housing dataset
        
    Returns:
        Dict[str, List[str]]: Dictionary with feature categories and their columns
    """
    feature_categories = {
        'ORIGINAL_NUMERICAL': ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 
                               'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 
                               'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 
                               'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 
                               'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 
                               'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 
                               'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 
                               'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SalePrice'],
        
        'ORIGINAL_CATEGORICAL': ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 
                                'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 
                                'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 
                                'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 
                                'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 
                                'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
                                'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 
                                'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 
                                'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 
                                'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 
                                'SaleType', 'SaleCondition'],
        
        'ENGINEERED_FEATURES': ['TotalSF', 'TotalBath', 'HouseAge', 'YearsSinceRemodel', 
                               'TotalPorchSF', 'QualitySize', 'QualityLivArea', 
                               'BathBedRatio', 'LivAreaRatio', 'PricePerSF'],
        
        'BINARY_INDICATORS': ['Has2ndFloor', 'HasBasement', 'HasGarage', 'HasPool', 
                             'HasFireplace', 'HasWoodDeck', 'HasOpenPorch', 
                             'HasEnclosedPorch', 'HasScreenPorch', 'Has3SsnPorch'],
        
        'CATEGORICAL_GROUPS': ['AgeCategory', 'QualityTier', 'SizeCategory', 'NeighborhoodTier']
    }
    
    # Filter to only include features that exist in the dataset
    for category, features in feature_categories.items():
        feature_categories[category] = [f for f in features if f in df.columns]
    
    return feature_categories


if __name__ == "__main__":
    # Example usage
    df = get_analysis_ready_data()
    feature_summary = get_feature_summary(df)
    
    print("\n📊 Feature Summary:")
    for category, features in feature_summary.items():
        print(f"   {category}: {len(features)} features")
        if len(features) <= 10:  # Show features if not too many
            print(f"      {features}")
        else:
            print(f"      {features[:5]}... (+{len(features)-5} more)")
