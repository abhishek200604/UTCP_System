# 🚦 UTCP System — Urban Traffic Congestion Prediction

A Machine Learning web application that predicts urban traffic congestion in **Pune, India** using hourly sensor data from **20 city areas**.

> **Model Accuracy: R² = 1.0** — Powered by 21-feature engineering with condition-aware prediction.

🔗 **Live Project:** [utcpsystem.streamlit.app](https://utcpsystem.streamlit.app/)

---

## 📂 Project Structure

```
UTCP/
├── app.py                              # Streamlit dashboard (3 pages)
├── train_model.py                      # ML training pipeline (styled output)
├── setup.py                            # Dependency installer (styled output)
├── utils.py                            # Data processing & feature engineering
├── README.md                           # This file
├── data/
│   └── pune_january_20areas.csv        # Traffic dataset (14,881 rows)
└── models/
    ├── congestion_model.pkl            # Best trained model (joblib)
    ├── scaler.pkl                      # Feature scaler
    └── model_metrics.json              # Evaluation metrics (JSON)
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip

### Setup & Run

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd UTCP

# 2. Install dependencies (styled output)
python setup.py

# 3. Train the model
python train_model.py

# 4. Launch the dashboard
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 📊 Dataset Description

| Field                              | Description                                           |
| ---------------------------------- | ----------------------------------------------------- |
| `Date`, `Time`                     | Timestamp (hourly, Jan 2025)                          |
| `Area_Name`                        | One of 20 Pune areas (Shivaji Nagar, Hinjewadi, etc.) |
| `Latitude`, `Longitude`            | Geo-coordinates for mapping                           |
| `Traffic_Volume`                   | Vehicle count per hour                                |
| `Avg_Speed_kmph`                   | Average speed on road                                 |
| `Traffic_Score`                    | Raw congestion score (0–100)                          |
| `Congestion_Level`                 | Low / Medium / High / Severe                          |
| `Weather`                          | Clear / Cloudy / Fog / Rain                           |
| `Accident`, `Road_Work`            | Yes / No binary indicators                            |
| `Event_Name`, `Event_Impact_Level` | Name and impact (None/Low/Medium/High)                |
| `Holiday_Traffic_Score`            | Adjusted score on holidays                            |
| `Final_Congestion_Score`           | **Target variable** (0–100)                           |

### Realistic Hourly Traffic Patterns

The data pipeline applies Pune-calibrated hourly congestion multipliers to simulate real urban traffic:

| Time Window          | Avg Score   | Level         |
| -------------------- | ----------- | ------------- |
| 🌙 Night (0–5 AM)    | 12 – 22     | Low           |
| 🌅 Morning (6–7 AM)  | 35 – 51     | Medium        |
| 🚗 **Rush (8–9 AM)** | **72 – 77** | **Severe**    |
| ☀️ Midday (10–14)    | 49 – 64     | Medium / High |
| 🚗 **Rush (17–18)**  | **78 – 82** | **Severe**    |
| 🌆 Evening (19–21)   | 47 – 76     | High          |
| 🌙 Late (22–23)      | 26 – 34     | Low / Medium  |

---

## 🤖 ML System

### Feature Engineering (21 Features)

| Category            | Features                                                             |
| ------------------- | -------------------------------------------------------------------- |
| **Temporal**        | Hour, DayOfWeek, DayOfMonth, Is_Weekend, Is_Holiday                  |
| **Traffic**         | Traffic_Volume, Avg_Speed_kmph, Traffic_Score, Holiday_Traffic_Score |
| **Categorical**     | Weather, Event_Impact, Congestion_Num, Accident, Road_Work           |
| **Spatial**         | Area_Encoded, Area_Congestion_Index, Latitude, Longitude             |
| **Spatio-temporal** | Rolling_Congestion_3h, Area_Hour, Holiday_Boost                      |

### Model Performance

| Model                         | R²      | MAE    | RMSE   |
| ----------------------------- | ------- | ------ | ------ |
| **Linear Regression** ⭐      | **1.0** | 0.0000 | 0.0000 |
| Gradient Boosting (500 trees) | 1.0     | 0.0382 | 0.1103 |
| Random Forest (500 trees)     | 0.9999  | 0.0746 | 0.1773 |

### Prediction Interface

The prediction interface uses **condition-aware progressive filtering**:

- Instead of simple area+hour averages, it filters historical data by the user's selected **accident, road work, weather,** and **event impact** conditions
- Progressively relaxes filters until matching rows are found
- Produces accurate predictions for any scenario combination

### Pipeline

1. Load CSV → parse datetime → clean typos
2. Apply realistic hourly congestion multipliers (Pune calibrated)
3. Smooth noise via 5-hour rolling window
4. Engineer 21 features
5. Normalize with MinMaxScaler
6. Train/test split (80/20)
7. Train & compare 3 models → save best

---

## 📡 Sparsity & Noise Handling

| Technique                | Description                               |
| ------------------------ | ----------------------------------------- |
| **Sparse simulation**    | Randomly remove 10–50% of rows            |
| **Noise injection**      | Add Gaussian noise to speed & volume      |
| **Smoothing**            | Rolling-mean over 5-hour windows per area |
| **Reconstruction**       | Linear interpolation + median fallback    |
| **Degradation analysis** | Retrain under degraded data, plot R² drop |

---

## 🏗️ Dashboard Pages

| Page                            | Features                                                                    |
| ------------------------------- | --------------------------------------------------------------------------- |
| 🔮 **Prediction Interface**     | Condition-aware prediction with gauge visualization, grouped input sections |
| 🗺️ **Pune Traffic Map**         | Interactive dark-themed scatter map with 4-level congestion-colored markers |
| 📈 **Spatio-Temporal Analysis** | Hour×Area heatmap, area comparison bar chart, hourly congestion trend line  |

### UI Controls

| Control                                  | Type                                 | Used In         |
| ---------------------------------------- | ------------------------------------ | --------------- |
| Hour selection                           | Number input (1–12) + AM/PM dropdown | Prediction, Map |
| Day of Week                              | Horizontal radio pills (Mon–Sun)     | Prediction      |
| Day of Month                             | Number input with +/- stepper        | Prediction      |
| Area                                     | Searchable dropdown (20 Pune areas)  | Prediction      |
| Weather                                  | Horizontal radio pills               | Prediction      |
| Event Impact                             | Horizontal radio pills               | Prediction      |
| Holiday / Weekend / Accident / Road Work | Toggle switches                      | Prediction      |

### UI Features

- **UTCP System** branded sidebar with styled navigation buttons
- Dual-theme CSS (light & dark mode compatible)
- Grouped input sections with bordered containers (Location & Time · Conditions · Situational Flags)
- Responsive layout — columns stack on mobile (≤768px), charts scale dynamically
- Colored ANSI terminal output for training & setup scripts
- Interactive Plotly charts with auto-margins and responsive rendering

---

## ☁️ Deployment

The app is deployed on **Streamlit Cloud** and accessible at:

🔗 **[utcpsystem.streamlit.app](https://utcpsystem.streamlit.app/)**

### Deploy Your Own

1. Push the project to a **GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set **Main file**: `app.py`
4. Click **Deploy**

> Make sure `models/` folder is committed (run `python train_model.py` before pushing).

---

## 🛠️ Technologies

| Technology     | Purpose                    |
| -------------- | -------------------------- |
| Python 3.10+   | Core language              |
| Pandas / NumPy | Data processing            |
| Scikit-Learn   | ML models & metrics        |
| Streamlit      | Web dashboard framework    |
| Plotly         | Interactive visualizations |
| Joblib         | Model serialization        |

---

## 📜 License

This project is for **academic and educational purposes**.
