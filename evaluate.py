# ==========================================
# MOVIE RATING PREDICTION
# Model Evaluation Program
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# ==========================================
# 1. LOAD MODEL AND DATA
# ==========================================

print("Loading trained model...")
model = joblib.load("models/movie_rating_model.pkl")

print("Loading dataset...")
df = pd.read_csv("dataset/movies_metadata.csv")

# ==========================================
# 2. DATA PREPROCESSING (same as train_model.py)
# ==========================================

numeric_columns = [
    "budget",
    "popularity",
    "revenue",
    "runtime",
    "vote_average",
    "vote_count"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.drop_duplicates()

for column in [
    "budget",
    "popularity",
    "revenue",
    "runtime",
    "vote_count"
]:
    df[column] = df[column].fillna(df[column].median())

# Extract features and target variable
feature_columns = [
    "budget",
    "popularity",
    "revenue",
    "runtime",
    "vote_count"
]

# Add release_year if available
if "release_date" in df.columns:
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year
    df["release_year"] = df["release_year"].fillna(df["release_year"].median())
    feature_columns.append("release_year")

X = df[feature_columns].copy()
y = df["vote_average"].copy()

# Remove rows with NaN values
valid_indices = ~(X.isnull().any(axis=1) | y.isnull())
X = X[valid_indices]
y = y[valid_indices]

print(f"Dataset shape after preprocessing: {X.shape}")

# ==========================================
# 3. MODEL EVALUATION
# ==========================================

print("\n" + "="*50)
print("MODEL EVALUATION METRICS")
print("="*50)

# Make predictions
y_pred = model.predict(X)

# Calculate metrics
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print(f"\nMean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# ==========================================
# 4. GENERATE EVALUATION RESULTS
# ==========================================

results_df = pd.DataFrame({
    'Metric': ['MAE', 'MSE', 'RMSE', 'R² Score'],
    'Value': [mae, mse, rmse, r2]
})

print("\nEvaluation Results Summary:")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv("outputs/model_results.csv", index=False)
print("\nResults saved to outputs/model_results.csv")

# ==========================================
# 5. VISUALIZATION
# ==========================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Actual vs Predicted
axes[0, 0].scatter(y, y_pred, alpha=0.5, edgecolors='k')
axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
axes[0, 0].set_xlabel("Actual Ratings")
axes[0, 0].set_ylabel("Predicted Ratings")
axes[0, 0].set_title("Actual vs Predicted Ratings")
axes[0, 0].grid(True, alpha=0.3)

# 2. Residuals
residuals = y - y_pred
axes[0, 1].scatter(y_pred, residuals, alpha=0.5, edgecolors='k')
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel("Predicted Ratings")
axes[0, 1].set_ylabel("Residuals")
axes[0, 1].set_title("Residual Plot")
axes[0, 1].grid(True, alpha=0.3)

# 3. Residuals Distribution
axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel("Residuals")
axes[1, 0].set_ylabel("Frequency")
axes[1, 0].set_title("Distribution of Residuals")
axes[1, 0].grid(True, alpha=0.3)

# 4. Metrics Bar Chart
metrics_names = ['MAE', 'RMSE', 'R² Score']
metrics_values = [mae, rmse, r2]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
axes[1, 1].bar(metrics_names, metrics_values, color=colors, edgecolor='black', alpha=0.7)
axes[1, 1].set_ylabel("Value")
axes[1, 1].set_title("Model Performance Metrics")
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("outputs/eda/model_evaluation.png", dpi=300, bbox_inches='tight')
print("\nVisualization saved to outputs/eda/model_evaluation.png")

print("\n" + "="*50)
print("EVALUATION COMPLETE")
print("="*50)
