# Housing Prices Dataset Documentation

## Dataset Overview

**Dataset Name:** Ames Housing Dataset  
**Source:** Kaggle - House Prices: Advanced Regression Techniques  
**File Location:** `./house-prices-advanced-regression-techniques/train.csv`  
**Dataset Type:** Supervised Learning (Regression)

### Basic Statistics
- **Total Observations:** ~1,460 houses
- **Total Features:** ~80 features
- **Target Variable:** SalePrice (Sale price in dollars)
- **Feature Types:** Mix of numerical and categorical

## Target Variable

**SalePrice** - The property's sale price in dollars
- **Type:** Continuous numerical
- **Purpose:** This is the target variable for prediction
- **Range:** Varies from affordable to luxury homes

## Feature Categories

### Numerical Features (~38 features)
Features that contain numeric values representing measurements, counts, or years.

**Examples:**
- `LotArea` - Lot size in square feet
- `YearBuilt` - Original construction year
- `GrLivArea` - Above grade living area square feet
- `BedroomAbvGr` - Number of bedrooms above grade
- `TotalBsmtSF` - Total basement square feet

### Categorical Features (~43 features)
Features that contain categories or labels.

**Examples:**
- `Neighborhood` - Physical location within Ames city limits
- `HouseStyle` - Style of dwelling
- `OverallQual` - Overall material and finish quality
- `BldgType` - Type of dwelling
- `MSZoning` - General zoning classification

## Complete Feature List

(Note: Refer to `data_description.txt` for detailed descriptions of all features)

### Property Identification
- `Id` - Observation number

### Property Characteristics
**Size & Space:**
- `LotArea`, `LotFrontage`, `GrLivArea`, `TotalBsmtSF`, `1stFlrSF`, `2ndFlrSF`

**Rooms & Facilities:**
- `BedroomAbvGr`, `KitchenAbvGr`, `TotRmsAbvGrd`, `FullBath`, `HalfBath`

**Quality & Condition:**
- `OverallQual`, `OverallCond`, `ExterQual`, `ExterCond`, `KitchenQual`

**Temporal:**
- `YearBuilt`, `YearRemodAdd`, `YrSold`, `MoSold`

**Location:**
- `Neighborhood`, `MSZoning`

**Construction:**
- `Foundation`, `RoofStyle`, `RoofMatl`, `Exterior1st`, `Exterior2nd`

**Amenities:**
- `Fireplaces`, `GarageCars`, `GarageArea`, `PoolArea`, `WoodDeckSF`

(Full detailed list with descriptions will be added as analysis progresses)

## Data Quality Notes

### Initial Observations
- Dataset appears complete with standard residential property features
- Mix of property characteristics, quality assessments, and amenities
- Temporal features allow for trend analysis over time
- Location features enable geographic price analysis

### Features of Interest
Key features expected to have strong correlation with house prices:
1. Overall Quality
2. Living Area (Square Footage)
3. Neighborhood
4. Year Built / Age
5. Garage Capacity
6. Basement Size

(Detailed analysis will be conducted in subsequent tasks)

## Usage Notes

This dataset is intended for:
- Regression modeling (predicting SalePrice)
- Feature engineering and selection
- Understanding factors that influence house prices
- Practice with real-world messy data (missing values, outliers, etc.)

## References

- **Kaggle Competition:** https://www.kaggle.com/c/house-prices-advanced-regression-techniques
- **Original Data:** Dean De Cock, Ames Iowa Assessor's Office

---

*Last Updated: October 24, 2025*
*This document will be updated as analysis progresses*
