# SalesForecasting_GOWRU-DEVI-VARA-VENKATA-SAI
# Demand Intelligence Engine 📈📦

An end-to-end predictive analytics and machine learning pipeline designed to optimize supply chain inventory, track anomalies, and segment product categories using internal transactional data and macroeconomic correlation features.

## 🚀 Core Features

### 1. Predictive Demand Forecaster
* **Dynamic Time-Series Forecasts:** Multi-horizon forecasting (1, 2, and 3-month ahead trajectories) using advanced model vectors tailored per product category.
* **Rigorous Validation Metrics:** Evaluated continuously using dynamic backtesting, achieving a Validation Mean Absolute Error (MAE) of **$1,831.10** and a Root Mean Squared Error (RMSE) of **$2,381.12** for core categories.

### 2. Time-Series Decomposition
* **Trend & Seasonality Isolation:** Extracts long-term underlying growth trajectories, isolates high-frequency seasonal spikes, and separates residual noise to improve downstream forecast stability.

### 3. Dual-Method Anomaly Detection Engine
Evaluates structural and behavioral shifts in weekly sales volume by comparing statistical and machine learning methods:
* **Isolation Forest Anomaly Classifier:** Successfully maps non-linear, multidimensional outlier patterns, catching sudden market swings and localized volatility clusters.
* **Z-Score Flagging (>2 Standard Deviations):** Configured as a baseline global metric to test statistical variance thresholds against dynamic trends.

### 4. Product Sub-Category Segments via PCA
Leverages Principal Component Analysis (PCA) and K-Means clustering to classify the product catalog into 4 operational archetypes for targeted warehouse allocation:
* **Cluster 0 (High-Velocity Outliers):** Extreme revenue-driving, massive-volume movers requiring high-priority bespoke monitoring.
* **Cluster 1 (Highly Volatile / Seasonal Items):** Trend-driven sub-categories requiring responsive, dynamic safety stock scaling.
* **Cluster 2 (Core Staples):** Highly dense, low-variance baseline items managed through automated continuous replenishment (ROP).
* **Cluster 3 (Mid-Volume Swing Items):** Moderately high-volume products subject to distinct demand fluctuations requiring hybrid buffer guardrails.

---

## 🛠️ Tech Stack & Architecture

* **Language:** Python 3.8+
* **Data Engineering & Analytics:** Pandas, NumPy, Scikit-learn (Isolation Forest, PCA, K-Means)
* **Statistical Modeling:** Statsmodels (Seasonal-Trend Decomposition using LOESS)
* **Visualization:** Matplotlib, Seaborn
* **Interface / Deployment:** Streamlit (Hosted via local/private cloud vectors)

---

## 📂 Repository Structure

```text
├── data/                  # Transactional historical records & macro features
├── notebooks/             # Exploratory analysis & algorithm prototyping
├── ├── analyisis.ipynb
├── ├── Charts
│   ├── ├── anomaly_detection.png
│   ├── ├──  market_correlation_matrix.png
│   └── └── pca_segmentation.png
├── app.py                 # Streamlit UI dashboard application
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
