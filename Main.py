import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

# File names for saving/loading trained model & preprocessing pipeline
MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

# Function to build preprocessing pipeline
def build_pipeline(num_attribs, cat_attribs):

    # Numeric pipeline: fill missing values with median, then scale
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: one-hot encode categories
    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine numeric + categorical transformations
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])
    
    return full_pipeline

# Train model if not already saved
if not os.path.exists(MODEL_FILE) or not os.path.exists(PIPELINE_FILE):
    housing = pd.read_csv('housing.csv')   # Load dataset

    # Create income categories for stratified sampling
    housing['income_cat'] = pd.cut(housing['median_income'],
                                   bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
                                   labels=[1, 2, 3, 4, 5])

    # Stratified split (train/test based on income_cat)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing, housing['income_cat']):
        # Save test set as input.csv for inference later
        housing.loc[test_index].drop("income_cat", axis=1).to_csv("input.csv", index=False)
        # Keep training set
        housing = housing.loc[train_index].drop("income_cat", axis=1)

    # Separate labels and features
    housing_labels = housing['median_house_value'].copy()
    housing_features = housing.drop('median_house_value', axis=1)

    # Identify numeric and categorical attributes
    num_attribs = housing_features.drop(['ocean_proximity'], axis=1).columns.tolist()
    cat_attribs = ['ocean_proximity']

    # Build pipeline and preprocess training data
    pipeline = build_pipeline(num_attribs, cat_attribs)
    housing_prepared = pipeline.fit_transform(housing_features)

    # Train Random Forest model
    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)

    # Save model and pipeline for future inference
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print("Model is trained. CONGRATS!")

# Inference (if model already exists)
else:
    # Load model and pipeline
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    # Read test input data
    input_data = pd.read_csv("input.csv")
    # Transform input data using saved pipeline
    transformed_input = pipeline.transform(input_data)
    # Predict housing prices
    predictions = model.predict(transformed_input)
    # Add predictions as a new column
    input_data['median_house_value'] = predictions

    # Save predictions to CSV
    input_data.to_csv("output.csv", index=False)

    print("INFERENCE COMPLETE.")
