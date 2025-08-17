import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# Load data again
# -----------------------------
housing = pd.read_csv("housing.csv")

# Features and labels
housing_labels = housing["median_house_value"].copy()
housing_features = housing.drop("median_house_value", axis=1)

# Reload pipeline (to ensure preprocessing is consistent)
pipeline = joblib.load("pipeline.pkl")
housing_prepared = pipeline.transform(housing_features)

# -----------------------------
# 1. Model Performance Comparison
# -----------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42)
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=5)
    rmse_scores = np.sqrt(-scores)
    results[name] = rmse_scores.mean()

# Plot comparison
plt.figure(figsize=(7, 5))
plt.bar(results.keys(), results.values(), color=["skyblue", "lightgreen", "orange"])
plt.ylabel("RMSE (lower is better)")
plt.title("Model Performance Comparison")
plt.savefig("model_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------
# 2. Feature Importance (Random Forest)
# -----------------------------
# Train Random Forest (or load from model.pkl)
forest_reg = joblib.load("model.pkl")

# Get feature names from pipeline
num_attribs = housing_features.drop(["ocean_proximity"], axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]
cat_encoder = pipeline.named_transformers_["cat"].named_steps["onehot"]
cat_one_hot = list(cat_encoder.get_feature_names_out(cat_attribs))

feature_names = num_attribs + cat_one_hot
importances = forest_reg.feature_importances_

# Sort by importance
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=90)
plt.ylabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300, bbox_inches='tight')
plt.show()
