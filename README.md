# Autovit Car Price Prediction Project

This project focuses on predicting used car prices on the Romanian market (Autovit) using various Machine Learning regression techniques. The workflow includes extensive Exploratory Data Analysis (EDA), domain-specific data cleaning (specifically for Electric Vehicles), feature engineering, and model comparison.

## Dataset and Exploratory Data Analysis (EDA)

The dataset consists of a complex mixture of numerical data (e.g., Km, Power), categorical data (e.g., Brand, Fuel), and unstructured text (e.g., equipment lists).

### Key Findings

* **Missing Data:** A diagnostic plot revealed critical missing values (over 90%) for columns like "Manufacturer Warranty" and "Battery Capacity," while essential attributes like "CO2 Emissions" had moderate missing values requiring imputation.
* **Target Variable (Price):** The initial price distribution was highly right-skewed with a skewness of 5.01. A logarithmic transformation (np.log1p) was applied, normalizing the distribution (skewness reduced to 0.13) and stabilizing error variance.
* **Correlations:** The strongest positive predictors for price were Year of Fabrication and Power, while Km had a strong negative correlation.

## Preprocessing and Feature Engineering

### 1. Encoding and Parsing

* **Boolean Mapping:** Text attributes (Yes/No) were converted to binary 1/0.
* **Text Extraction:** Regex was used to extract numeric values from text fields like "Range" and "Battery Capacity".
* **Feature Engineering:** Long lists of equipment (e.g., "Audio and Technology") were converted into a single numeric predictor: Number of Features. This confirmed the hypothesis that more features correlate with a higher price.
* **One-Hot Encoding:** Applied to categorical variables like Brand and Fuel.

### 2. Smart Imputation (EV Logic)

A specific logic was applied to Electric Vehicles (EVs). Many EVs had missing CO2 emissions data. By validating against battery capacity, these missing values were logically filled as CO2 = 0.

### 3. Feature Selection and Scaling

* **Multicollinearity Removal:** Attributes like "Cylindrical Capacity" were removed due to high correlation (0.77) with "Power".
* **Scaling:** StandardScaler was applied to all numeric attributes to normalize ranges (e.g., Year vs. Km).

## Modeling Strategy

The project compared four primary algorithms to establish a baseline and find the optimal regressor.

1. **Linear Ridge Regression (Baseline):** Established a strong baseline but had the highest error, indicating non-linear relationships.
2. **Random Forest Regressor:** A robust model, but slightly underfitted due to restrictive leaf constraints (min_samples_leaf=19).
3. **Support Vector Regressor (SVR):** Outperformed Random Forest using an RBF kernel to model non-linear price spaces.
4. **Gradient Boosting Regressor:** The best performing model. It utilized a depth of 7 to capture complex interactions.

## Results

The Gradient Boosting Regressor was the winning model, being the only one to surpass the 0.90 R-squared threshold.

| Rank | Algorithm | Optimal Parameters (Summary) | R2 Score | MAE (Euro) | MSE |
| --- | --- | --- | --- | --- | --- |
| 1 | **Gradient Boosting** | lr=0.1, depth=7, est=234 | **0.9101** | **3,495** | **7.72e+07** |
| 2 | GBR Quantile (Median) | alpha=0.5, est=200 | 0.8710 | 3,849 | - |
| 3 | SVR (Tuned) | C=31.4, kernel='rbf' | 0.8687 | 3,859 | 1.13e+08 |
| 4 | Random Forest | est=287, leaf=19 | 0.8454 | 4,446 | 1.33e+08 |
| 5 | Linear Ridge | alpha=98.8 | 0.8178 | 4,721 | 1.56e+08 |

### Uncertainty Analysis

A Quantile Regression model (Gradient Boosting with loss='quantile') was trained to generate a 90% prediction interval. The results showed heteroscedasticity: prediction bands are narrow for cheaper cars but widen significantly for luxury vehicles, reflecting real-market volatility.

## Visualizations

The visual representations for the following plots can be found in the PDF Report Documentation:

1. **Missing Values Plot:** Identifies columns requiring removal vs. imputation.
2. **Price Distribution:** Shows the impact of log-transformation on normalizing the target variable.
3. **Correlation Heatmap:** Highlights the top 15 features correlated with price.
4. **Prediction Intervals:** Visualizes the confidence range for the first 50 validation samples.
