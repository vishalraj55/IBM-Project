# 🚗 Used Car Price Prediction

A full-stack machine learning web app that predicts used-car selling prices.

Dataset: **Vehicle dataset from Cardekho** (Kaggle) — https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho

- **Backend:** Flask REST API
- **Frontend:** Streamlit
- **ML Model:** Linear Regression (scikit-learn), one-hot + scaled features in a `Pipeline`
- **Dataset:** `data/car_data.csv` (60 synthetic listings)

## Project Structure
```
car_price_project/
├── data/
│   └── car_price.csv         # Dataset
├── model/
│   ├── model.pkl              # Trained pipeline (generated)
│   ├── metrics.json           # Eval metrics (generated)
│   └── diagnostics.png        # Predicted vs actual plot (generated)
├── backend/
│   └── app.py                 # Flask REST API
├── frontend/
│   └── ui.py                  # Streamlit UI
├── train_model.py             # Model training script
├── requirements.txt
└── README.md
```

## Quickstart

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Train the model
```
python train_model.py
```
Saves `model/model.pkl`, `model/metrics.json`, `model/diagnostics.png`.

### 3. Start the Flask backend
```
python backend/app.py
```
API runs at `http://localhost:5000`

### 4. Launch the Streamlit frontend
(open a second terminal)
```
streamlit run frontend/ui.py
```
UI opens at `http://localhost:8501`

## API Endpoints

| Method | Endpoint      | Description                     |
|--------|---------------|----------------------------------|
| GET    | `/health`     | Health check                     |
| POST   | `/predict`    | Predict car price                |
| GET    | `/dataset`    | Return full dataset as JSON      |
| GET    | `/model_info` | Model metrics + feature list     |

### `POST /predict` — example
```json
// Request
{
  "age_years": 5,
  "mileage_km": 60000,
  "engine_cc": 1500,
  "owner_count": 1,
  "fuel_type": "Petrol",
  "transmission": "Manual"
}

// Response
{
  "predicted_price": 1327966.56,
  "currency": "INR",
  "input": { ... }
}
```

## Frontend Pages

| Page              | Description                                              |
|-------------------|-----------------------------------------------------------|
| Predict Price     | Input car details, get instant price + gauge chart        |
| Dataset Explorer  | Browse data, scatter/histogram/box/heatmap charts          |
| Model Insights    | R², MAE, RMSE, feature list, price-vs-age trendline        |

## Model Performance (actual, from this dataset)

| Metric | Value        |
|--------|--------------|
| R²     | ~0.90        |
| MAE    | ~₹96,000     |
| RMSE   | ~₹115,500    |

## Tech Stack

| Layer    | Technology                        |
|----------|------------------------------------|
| ML       | scikit-learn, numpy                |
| Backend  | Flask, flask-cors                  |
| Frontend | Streamlit, Plotly                  |
| Data     | pandas, matplotlib                 |
