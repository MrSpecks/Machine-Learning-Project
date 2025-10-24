# 🏠 Housing Prices: Advanced Regression Techniques
## Refactored Exploratory Data Analysis & Dashboard Backend

## 📊 Project Overview
This project provides a **refactored, production-ready** analysis of the Kaggle Housing Prices dataset. The analysis has been restructured into modular components suitable for dashboard integration while maintaining all original insights and findings.

**Project Type:** Machine Learning Developer Intern Technical Assessment  
**Domain:** Real Estate / Housing Market Analysis  
**Objective:** Predict house sale prices using advanced data science techniques  
**Version:** 2.0 (Refactored for Dashboard Backend)

## 🗂️ Dataset Information

### Primary Dataset
- **Source:** [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- **Location:** `./house-prices-advanced-regression-techniques/train.csv`
- **Size:** 1,460 observations × 80+ features (after engineering)
- **Target Variable:** SalePrice (continuous, $34,900 - $755,000)
- **Dataset Type:** Supervised Learning (Regression)

### Additional Files
- `test.csv` - Test dataset for predictions
- `data_description.txt` - Detailed feature descriptions
- `sample_submission.csv` - Submission format

## 🚀 Quick Start

### Prerequisites
- **Python:** 3.8 or higher
- **Jupyter:** Notebook or JupyterLab
- **Memory:** Minimum 4GB RAM recommended
- **Storage:** 100MB free space

### Installation Steps
1. **Clone/Download Repository:**
   ```bash
   git clone https://github.com/MrSpecks/Machine-Learning-Project
   cd Machine-Learning-Project
   ```

2. **Install Required Libraries:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Installation:**
   ```bash
   python -c "import pandas, numpy, matplotlib, seaborn; print('✅ All libraries installed successfully!')"
   ```

## ▶️ Running the Refactored Analysis

### Main Refactored Notebook
1. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

2. **Open Refactored Analysis:**
   - Navigate to `Housing_EDA_Refactored.ipynb`
   - Run all cells sequentially (Cell → Run All)

3. **Expected Runtime:** 3-5 minutes for complete analysis

### Using the Modular Components

#### Data Loading and Feature Engineering
```python
from housing_data_loader import get_analysis_ready_data, get_feature_summary

# Load the complete analysis-ready dataset
df = get_analysis_ready_data("house-prices-advanced-regression-techniques/train.csv")

# Get feature categories
feature_summary = get_feature_summary(df)
```

#### Analysis and Metrics
```python
from housing_analysis_metrics import (
    get_top_correlations, calculate_neighborhood_stats, 
    filter_dataset, get_price_distribution_stats
)

# Get top price drivers
top_correlations = get_top_correlations(df, n_top=10)

# Calculate neighborhood statistics
neighborhood_stats = calculate_neighborhood_stats(df)

# Filter dataset for dashboard
filtered_df = filter_dataset(df, {
    'Neighborhood': ['Northridge', 'Stone Brook'],
    'OverallQual': (7, 10)
})
```

## 📁 Project Structure
```
Machine-Learning-Project/
├── house-prices-advanced-regression-techniques/
│   ├── train.csv                    # Original training dataset
│   ├── test.csv                     # Test dataset
│   ├── train_cleaned.csv            # Cleaned dataset (EDA output)
│   ├── train_engineered.csv         # Enhanced dataset (Feature Engineering output)
│   ├── data_description.txt         # Feature descriptions
│   └── sample_submission.csv        # Submission format
├── housing_data_loader.py           # 🔧 Data loading and feature engineering module
├── housing_analysis_metrics.py      # 📊 Business logic and calculations module
├── Housing_EDA_Refactored.ipynb    # 📓 Refactored analysis notebook
├── housing_eda_kagiso_mfusi.ipynb       # Original EDA notebook
├── housing_feature_engineering_kagiso_mfusi.ipynb # Original feature engineering
├── DATASET_INFO.md                  # Detailed dataset documentation
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
└── prompts/                        # Documentation prompts
    ├── Convert-to-dasboard.txt
    └── structure-and-documentation-of-notebook.txt
```

## 🎯 Key Findings Summary

### 📈 **Primary Price Drivers**
• **OverallQual** (r=0.79) - Quality dominates pricing over size
• **GrLivArea** (r=0.71) - Living area is the strongest size predictor
• **GarageCars** (r=0.64) - Garage capacity more predictive than bedroom count
• **Location** - Neighborhood creates 3x price variation ($100k-$300k range)

### 🔍 **Surprising Insights**
• **Quality > Size** - OverallQual beats GrLivArea in importance hierarchy
• **All Outliers High-Value** - No low-value outliers, only luxury homes identified
• **Bedroom Count Weak** - BedroomAbvGr only r=0.17 correlation with price
• **Age Plateau Effect** - Depreciation stops after 50 years, historic homes maintain value

### 📊 **Data Quality Excellence**
• **Zero Missing Values** - All 19 features with missing data successfully imputed
• **No Duplicates** - Clean dataset with unique property records
• **Multicollinearity Documented** - High correlations identified for modeling consideration
• **Outliers Retained** - Legitimate luxury properties preserved for analysis

### 🏗️ **Feature Engineering Success**
• **50+ Features Created** - Including TotalSF, TotalBath, HouseAge, QualitySize interactions
• **High-Impact Features** - QualitySize, QualityLivArea show strong predictive power
• **Expected Performance Gain** - R² improvement from 0.85 to 0.90+ with engineered features

### 💼 **Business Insights**
• **Sellers:** Invest in quality renovations over size expansion for maximum ROI
• **Buyers:** Don't overpay for square footage alone - quality and location matter more
• **Agents:** Use neighborhood-specific comparables for accurate pricing
• **Developers:** Location first, then balance quality and size for optimal market positioning

### 🤖 **Modeling Readiness**
• **Clean Dataset** - Ready for immediate modeling with zero preprocessing needed
• **Feature-Rich** - 130+ total features (80 original + 50+ engineered) available
• **Validation Complete** - All features tested for correlation and multicollinearity
• **Expected R² Performance** - 0.85-0.92 range achievable with proper model selection

## 🔧 Refactoring Improvements

### **Modular Architecture**
- **`housing_data_loader.py`**: Centralized data loading, cleaning, and feature engineering
- **`housing_analysis_metrics.py`**: Business logic functions for calculations and insights
- **Refactored Notebook**: Clean, documented analysis workflow

### **Dashboard Integration Ready**
- **Filtering Functions**: `filter_dataset()` supports dynamic data filtering
- **Metric Functions**: All analysis functions work with filtered datasets
- **Consistent API**: Standardized function signatures and return types

### **Production Quality**
- **Comprehensive Documentation**: Docstrings for all functions
- **Type Hints**: Full type annotation for better IDE support
- **Error Handling**: Robust error checking and informative messages
- **Extensible Design**: Easy to add new features and analysis functions

## 📋 Requirements
```
# Core Data Science Libraries
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Statistical Analysis
scipy>=1.10.0

# Machine Learning
scikit-learn>=1.3.0

# Jupyter Environment
jupyter>=1.0.0
ipykernel>=6.0.0

# Optional: For enhanced dashboard development
streamlit>=1.28.0
plotly>=5.17.0
```

## 🚀 Streamlit Dashboard

### Running the Interactive Dashboard

1. **Install Dashboard Dependencies:**
   ```bash
   pip install streamlit plotly
   ```

2. **Launch the Dashboard:**
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Access the Dashboard:**
   - Open your browser to `http://localhost:8501`
   - Use the sidebar filters to explore the data interactively

### Dashboard Features

**📊 Key Performance Indicators (KPIs):**
- Median Sale Price with trend indicators
- Average Price per Square Foot
- Total Properties Sold
- Average Overall Quality

**📈 Interactive Charts:**
- **Price vs. Size Scatter Plot**: Living area vs. sale price (colored by quality)
- **Neighborhood Box Plots**: Price distribution by neighborhood
- **Quality Bar Chart**: Median prices by quality rating
- **Correlation Heatmap**: Key feature correlations with sale price

**🔍 Dynamic Filtering:**
- Neighborhood selection (multiselect)
- Overall Quality range (slider)
- Year Sold selection (multiselect)
- Price Range (slider)

**📋 Additional Statistics:**
- Average Property Age at Sale
- Average Garage Capacity

### Dashboard Integration Example

```python
from housing_data_loader import get_analysis_ready_data
from housing_analysis_metrics import (
    get_top_correlations, calculate_neighborhood_stats, 
    filter_dataset, get_price_distribution_stats
)

# Load data
df = get_analysis_ready_data()

# Apply user filters (e.g., from Streamlit dashboard)
filters = {
    'Neighborhood': ['Northridge', 'Stone Brook'],
    'OverallQual': (7, 10),
    'YearBuilt': (2000, 2010)
}
filtered_df = filter_dataset(df, filters)

# Get insights for filtered data
correlations = get_top_correlations(filtered_df)
neighborhood_stats = calculate_neighborhood_stats(filtered_df)
price_stats = get_price_distribution_stats(filtered_df)

# Use these metrics in your dashboard
```

## 👨‍💻 Author Information

**Full Name:** Kagiso Mfusi  
**Email:** Kagisomfusi@outlook.com  
**LinkedIn:** [https://www.linkedin.com/in/kagiso-mfusi-95b329224](https://www.linkedin.com/in/kagiso-mfusi-95b329224)  
**Project Date:** October 24, 2025 
**Assessment:** Machine Learning Developer Technical Assessment

## 📄 License
This project is created for educational and assessment purposes only. The dataset is sourced from Kaggle's public House Prices competition.
