# 📊 Sales Data Analysis & Prediction Project

## Overview
This project explores a Kaggle dataset containing sales records, transforming it into actionable insights through data visualization and predictive modeling. The ultimate goal is to identify high-value customers using machine learning.

## 🔍 Dataset
- Sourced from Kaggle
- Loaded and explored using `df.info()`, `df.describe()` and `df.hist()`
- Initial insights:
  - Data is skewed toward low sales values
  - High sales values inflate the average (highest: 22,638)

## 🧹 Data Preprocessing
- Converted date columns using `pd.to_datetime()`
- Created `Month`, `Year`, and `Month-Year` columns for time-based analysis
- Calculated time-to-ship: `Ship Date - Order Date`
- Verified newly created columns with sample outputs

## 📊 Exploratory Data Analysis

### Sales by Category
| Category         | Total Sales |
|------------------|-------------|
| Furniture        | 728,658.58  |
| Office Supplies  | 705,422.33  |
| Technology       | 827,455.87  |

*Visualized with graphs (see repo)*

### Sales by Sub-Category *(No graph included)*
| Sub-Category | Total Sales |
|--------------|-------------|
| Accessories  | 164,186.70  |
| Appliances   | 104,618.40  |
| Art          | 26,705.41   |
| Binders      | 200,028.79  |
| Bookcases    | 113,813.20  |
| Chairs       | 322,822.73  |
| Copiers      | 146,248.09  |
| Envelopes    | 16,128.05   |
| Fasteners    | 3,001.96    |
| Furnishings  | 89,212.02   |
| Labels       | 12,347.73   |
| Machines     | 189,238.63  |
| Paper        | 76,828.30   |
| Phones       | 327,782.45  |
| Storage      | 219,343.39  |
| Supplies     | 46,420.31   |
| Tables       | 202,810.63  |

## 🧠 Feature Engineering
- Counted total orders for each customer
- Created Boolean column for “frequent customers” (≥10 orders)
- Merged this data back into main dataframe

## 🎯 Prediction Task
- Defined `High Value` target based on sales thresholds
- Target distribution:
  - `False`: 77.9%
  - `True`: 22.1%

## ⚙️ Model Preparation
- Selected relevant features
- Encoded categorical variables
- Scaled numerical features to normalize ranges
- Split data into training and testing sets
- Trained initial predictive model (details in notebook)
