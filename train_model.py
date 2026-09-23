import json

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "data/car_data.csv"
CURRENT_YEAR = 2024

NUMERIC_FEATURES = ["car_age", "km_driven"]
CATEGORICAL_FEATURES = ["fuel", "seller_type", "transmission", "owner"]
TARGET = "selling_price"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates().dropna(subset=["year", "selling_price", "km_driven"])
    df["car_age"] = CURRENT_YEAR - df["year"]
    df = df[df["car_age"] >= 0]
    return df


def main():
    df = load_data()

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"R2:   {r2:.4f}")
    print(f"MAE:  {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")

    joblib.dump(model, "model/model.pkl")

    metrics = {
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "source": "https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho",
    }
    with open("model/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.6, edgecolor="k", s=25)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual Selling Price")
    ax.set_ylabel("Predicted Selling Price")
    ax.set_title(f"Predicted vs Actual (R2={r2:.2f})")
    fig.tight_layout()
    fig.savefig("model/diagnostics.png", dpi=120)

    print("Saved model/model.pkl, model/metrics.json, model/diagnostics.png")


if __name__ == "__main__":
    main()
