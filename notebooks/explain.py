# Car Price Prediction - Senior Data Science Notebook (XGBoost)
# **Project:** Used Vehicle Price Estimation
# **Model:** XGBoost Regressor
# **Goal:** Production-ready model artifact
# **Date:** 2025-12-31

import pandas as pd
import numpy as np
import datetime
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# 1. DATA LOADING
# ---------------------------------------------------------
print("[INFO] Loading data...")
df = pd.read_csv("data/raw/vehicles.csv")

# Standardize column names
df.columns = df.columns.str.lower()
print(f"[INFO] Raw data shape: {df.shape}")

# 2. DATA CLEANING & PREPROCESSING
# ---------------------------------------------------------
print("[INFO] Cleaning data...")

# Drop irrelevant columns (Added 'posting_date' which caused the crash)
drop_cols = ['url', 'region_url', 'image_url', 'lat', 'long',
             'vin', 'id', 'county', 'description', 'region', 'posting_date']
df = df.drop(columns=drop_cols, errors='ignore')

# Filter outliers (Price & Year)
# Range: $1,000 - $60,000
df = df[(df['price'] > 1000) & (df['price'] < 60000)]
# Range: 2005+
df = df[df['year'] > 2005]

# Handle Missing Values
# Categorical -> 'unknown'
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna('unknown')

# Numerical -> Mean
num_cols = df.select_dtypes(include=['number']).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

print(f"[INFO] Cleaned data shape: {df.shape}")

# 3. FEATURE ENGINEERING
# ---------------------------------------------------------
# Feature 1: Car Age
current_year = datetime.datetime.now().year
df['car_age'] = current_year - df['year']

# Feature 2: Model Simplification (Top 50 models)
top_models = df['model'].value_counts().head(50).index
df['model_new'] = df['model'].apply(lambda x: x if x in top_models else 'other')

# Drop original columns
df = df.drop(columns=['model', 'year'])

# 4. ENCODING
# ---------------------------------------------------------
le_dict = {}
# Explicitly listing categorical columns to ensure order matches
encode_cols = ['manufacturer', 'model_new', 'condition', 'cylinders', 'fuel',
               'title_status', 'transmission', 'drive', 'size', 'type',
               'paint_color', 'state']

for col in encode_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

# 5. MODEL TRAINING (XGBOOST)
# ---------------------------------------------------------
X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"[INFO] Training XGBoost Model on {X_train.shape[0]} samples...")

# XGBoost Parameters
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42
)

xgb_model.fit(X_train, y_train)
print("[INFO] Training complete.")

# 6. EVALUATION
# ---------------------------------------------------------
y_pred = xgb_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("-" * 30)
print("XGBOOST PERFORMANCE REPORT")
print("-" * 30)
print(f"RMSE     : {rmse:.2f}")
print(f"R2 Score : {r2:.4f}")
print("-" * 30)

# 7. SERIALIZATION
# ---------------------------------------------------------
joblib.dump(xgb_model, 'car_price_model.pkl')
print("[INFO] Model saved to 'car_price_model.pkl'")