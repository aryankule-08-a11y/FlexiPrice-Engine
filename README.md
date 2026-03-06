# ⚡ FlexiPrice Engine — Dynamic Price Optimization System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Plotly-5.18+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

<p align="center">
  An AI-powered pricing engine that maximizes e-commerce revenue by analyzing demand, competition, inventory, and market signals in real time.
</p>

---

## 🎯 Overview

**FlexiPrice Engine** is a production-grade Dynamic Price Optimization System built with Streamlit. It demonstrates how companies like Amazon, Uber, and airlines use machine learning to dynamically adjust prices based on real-time market conditions.

This project was built as an academic project for **Big Data / Data Science** coursework.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **ML Price Prediction** | Random Forest (R² = 0.9868) + Linear Regression models |
| 🎛️ **Interactive Controls** | Sliders & dropdowns for demand, inventory, traffic, season |
| 📊 **Rich Visualizations** | 15+ Plotly charts — scatter, line, bar, heatmap, pie |
| 💰 **Revenue Simulation** | Revenue curve with optimal price marker |
| 📉 **Price Elasticity** | Demand response visualization |
| 🔮 **Demand Forecasting** | 30-day simulated forecast with confidence bands |
| 🏷️ **Competitor Analysis** | Side-by-side price comparison by category |
| 💡 **Business Insights** | 6 strategic recommendation cards |
| 📥 **Export Results** | Download predictions & reports as CSV |
| 🎨 **Premium Dark UI** | Glassmorphism, gradients, animations |

---

## 🖥️ Dashboard Pages

### 🏠 Home
- Project introduction & KPI overview
- Dynamic pricing explanation
- "How It Works" pipeline diagram

### 📊 Data Analysis
- Dataset preview (2,000 rows × 15 columns)
- Price distribution histograms
- Demand vs Price scatter plots
- Category revenue comparison (bar + pie)
- Feature correlation heatmap

### 🤖 Pricing Model
- Model performance metrics (R², MAE, RMSE)
- Live price prediction with color-coded output
- Revenue simulation curve
- Profit estimation chart
- Actual vs Predicted comparison
- Feature importance ranking

### 💡 Insights
- Price elasticity of demand curve
- Competitor price comparison
- 30-day demand forecast
- Peak vs Off-Peak analysis
- Strategic business recommendations

---

## 🏗️ Project Structure

```
FlexiPrice-Engine/
│
├── app.py                  # Main Streamlit dashboard (4 pages)
├── model.py                # ML models (Random Forest + Linear Regression)
├── utils.py                # Utility functions, KPI cards, chart theming
├── generate_dataset.py     # Synthetic dataset generator
├── dataset.csv             # Generated e-commerce dataset
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── assets/
    └── style.css           # Custom premium dark-theme CSS
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/aryankule-08-a11y/FlexiPrice-Engine.git
cd FlexiPrice-Engine

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the dataset (optional — auto-generated on first run)
python generate_dataset.py

# 4. Launch the app
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 📊 Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| `product_id` | String | Unique product identifier |
| `product_category` | Categorical | Electronics, Fashion, Books, etc. |
| `base_price` | Float | Original listed price ($5–$1,200) |
| `competitor_price` | Float | Competitor's current price |
| `demand_level` | Integer | Demand index (5–100) |
| `inventory_level` | Integer | Units in stock (1–500) |
| `customer_traffic` | Integer | Daily visitor count (50–5,000) |
| `time_of_day` | Categorical | Morning / Afternoon / Evening / Night |
| `day_of_week` | Categorical | Monday – Sunday |
| `season` | Categorical | Spring / Summer / Autumn / Winter |
| `is_peak` | Binary | 1 = Peak period, 0 = Off-peak |
| `discount_pct` | Float | Applied discount (0%–35%) |
| `units_sold` | Integer | Units sold that day |
| `revenue` | Float | Daily revenue |
| `optimal_price` | Float | Target variable (ML-predicted) |

---

## 🤖 Model Performance

| Model | R² Score | MAE | RMSE |
|-------|----------|-----|------|
| **Random Forest** | 0.9868 | $26.08 | $53.52 |
| Linear Regression | 0.9763 | $30.14 | $62.87 |

---

## 🛠️ Tech Stack

- **Python** — Core programming language
- **Streamlit** — Web application framework
- **Pandas & NumPy** — Data manipulation
- **Scikit-learn** — Machine learning models
- **Plotly** — Interactive visualizations
- **Custom CSS** — Premium dark-theme styling

---

## 👤 Author

**Aryan Kule**
BSc Data Science • 2026

---

## 📄 License

This project is licensed under the MIT License.
