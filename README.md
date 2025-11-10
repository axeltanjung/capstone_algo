# Intelligent Ore Quality & Throughput Forecasting System (IOQTFS)
**End-to-End Machine Learning and Time Series Forecasting for Ore Classification, Composition, and Tonnage Prediction**

---

## 1. Overview

This project demonstrates a full **mine-to-mill analytics pipeline** that integrates **ore type classification**, **ore composition forecasting**, and **mill tonnage prediction**.

It connects upstream geological features with downstream processing performance — enabling data-driven control of blending, throughput, and cost efficiency.

### Key Components:
1. **Ore Type Classification** → Predicts ore type per geological sample.  
2. **Ore Composition Forecasting (Prophet)** → Forecasts daily % proportion of ore types feeding the mill.  
3. **Mill Tonnage Forecasting (Prophet)** → Projects total daily tonnage throughput considering ore composition and operational cycles.

---

## 2. Project Structure

**Folder layout:**

- `data/`
  - `ore_type_data.csv` — Raw ore sample characteristics (Mixed, Oxide, Sulphide)
  - `simulated_tonage.csv` — Daily mill throughput / tonnage

- `notebooks/`
  - `ore_type_classification_and_tonnage_forecasting.ipynb` — ML pipeline: Random Forest, Decision Tree and Time Series Forcaseting

- `models/`
  - `best_model_random_forest.pkl` — Trained ore type classification model
  - `best_model_knn.pkl` — Trained ore type classification model
  - `best_model_logistic_regression.pkl` — Trained ore type classification model
  - `best_model_decision_tree.pkl` — Trained ore type classification model

- `README.md` — Project documentation

---

## 🔬 3. Ore Type Classification

### Objective
Predict ore type (`Mixed`, `Oxide`, `Sulphide`) from geological and chemical features.

### Dataset Features

| Feature | Description |
|----------|--------------|
| `depth_mean` | Average sample depth (m) |
| `fe_pct` | Iron content (%) |
| `cu_pct` | Copper content (%) |
| `sio2_pct` | Silica percentage |
| `moisture_pct` | Moisture content (%) |
| `au_gpt` | Gold grade (g/t) |
| `pit` | Source pit name |
| `sample_id` | Unique sample identifier |
| **Target:** `ore_type_label` | Classified ore type |

### Models Tested
**- Random Forest ✅ (best performing)**
- Decision Tree
- Logistic Regression
- XGBoost

### Evaluation Metrics

| Metric | Purpose |
|---------|----------|
| Accuracy | Overall correctness |
| F1-Score | Balanced view for imbalanced ore distribution |
| Confusion Matrix | Type-specific error analysis |

### Output Utilization
Predicted labels are aggregated **daily** to calculate the **proportion (%) of each ore type** entering the mill — the key input for forecasting and blending optimization.

---

## 4. Ore Composition Forecasting (Prophet)

### Objective
Forecast the **daily proportion (%)** of each ore type (e.g., Mixed, Oxide, Sulphide) over time using Prophet.

### Modeling Setup

```python
m = Prophet(
    seasonality_mode='multiplicative',
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=2.0
)
m.add_regressor('oxide_lag1')
m.add_regressor('sulphide_lag1')
m.add_seasonality(name='weekly', period=7, fourier_order=3)
m.add_seasonality(name='yearly', period=365, fourier_order=8)
```

### Insights Captured

* **Trend:** Long-term compositional drift (e.g., oxide ore declining over time).
* **Seasonality:** Operational cycles or weekly mining routines.
* **Holiday Effects:** Maintenance shutdowns or production pauses.

### Metrics

| Metric | Description                 |
| ------ | --------------------------- |
| RMSE   | Absolute forecast deviation |
| MAPE   | Relative forecast error (%) |

---

## 5. Mill Tonnage Forecasting (Prophet)

### Objective

Predict **daily total mill throughput (tons/day)** based on historical tonnage and ore-type composition.

### Dataset Example

| Date       | mixed\_pct | oxide\_pct | sulphide\_pct | total\_tonnage |
| ---------- | --------- | --------- | ------------ | ------------- |
| 2023-01-01 | 0.42      | 0.35      | 0.23         | 98,500        |
| 2023-01-02 | 0.40      | 0.37      | 0.23         | 97,800        |
| ...        | ...       | ...       | ...          | ...           |

### Prophet Setup

```python
m_ton = Prophet(
    seasonality_mode='multiplicative',
    changepoint_prior_scale=0.1,
    seasonality_prior_scale=5.0
)
m_ton.add_regressor('mixed_pct')
m_ton.add_regressor('oxide_pct')
m_ton.add_regressor('sulphide_pct')
m_ton.add_seasonality(name='weekly', period=7, fourier_order=3)
```

### Metrics

| Metric | Description                            |
| ------ | -------------------------------------- |
| RMSE   | Measures tonnage forecast deviation    |
| MAPE   | Percent error of forecasted throughput |
| R²     | Goodness of fit (optional)             |

### Use Case

Forecasting mill tonnage helps operations:

* Anticipate production constraints
* Plan maintenance during low throughput
* Adjust blending for optimal recovery and energy use

---

## 6. Results & Figures (Placeholder)

This section is intended to visually display the model performance and forecasts.

| Figure | Description | File/Placeholder |
| :--- | :--- | :--- |
| Ore Classification | Confusion Matrix and Feature Importance | `ore_classification_metrics.png` |
| Composition Forecast | Predicted vs. Actual Daily Ore Proportions | `ore_composition_forecast.png` |
| Tonnage Forecast | 14-Day Ahead Tonnage Projection with Uncertainty Bounds | `mill_tonnage_forecast.png` |

---

## 7. Business Impact & Optimization Metrics

| Metric              | Description                           | Optimization Goal                    |
| ------------------- | ------------------------------------- | ------------------------------------ |
| Ore type proportion | % composition of Mixed/Oxide/Sulphide | Stable blending ratios               |
| Mill throughput     | Tons processed per day                | Maximize without overloading         |
| Energy consumption  | kWh per ton                           | Minimize via smoother feed           |
| Recovery rate       | % of valuable metal extracted         | Increase via consistent feed quality |
| Downtime            | Unplanned production halts            | Reduce variability impact            |

### Integration

Prophet forecasts → Operational dashboard → Early alerts for:

* Unbalanced ore feed (composition drift)
* Energy spikes due to ore hardness
* Throughput inefficiency caused by high sulphide concentration

---

## 8. Future Enhancements

* Integrate **cost modeling** (e.g., energy or reagent cost/ton).
* Extend Prophet with **exogenous regressors** for weather, shift, or equipment runtime.
* Deploy interactive dashboards with **Streamlit or Power BI**.
* Apply **reinforcement learning** for adaptive mill control based on forecast feedback.

---

## 9. Key Takeaways

This project illustrates the synergy of:

* **Classification (upstream)** → ore characterization
* **Forecasting (midstream)** → ore composition prediction
* **Throughput modeling (downstream)** → production and cost forecasting

It provides a framework for **data-driven decision-making** across the mine-to-mill value chain.

---

## 10. Tech Stack

| Layer             | Tools                 |
| ----------------- | --------------------- |
| Data Manipulation | Python, Pandas, NumPy |
| Machine Learning  | scikit-learn          |
| Time Series       | Facebook Prophet      |
| Visualization     | Matplotlib, Seaborn   |
| Deployment        | Streamlit, Power BI   |

---

## Author

**Axel Ivanda Tanjung**
*Data Scientist — Data Scientist*


