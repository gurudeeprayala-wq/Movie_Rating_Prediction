# ==========================================
# MOVIE RATING PREDICTION
# Main Training Program
# ==========================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import joblib
# ==========================================
# 1. LOAD DATASET
# ==========================================
df = pd.read_csv("dataset/movies_metadata.csv")

print("Dataset loaded successfully!")
# ==========================================
# 2. DATA UNDERSTANDING
# ==========================================

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\nMovie Rating Statistics:")
print(df["vote_average"].describe())

# ==========================================
# 3. EXPLORATORY DATA ANALYSIS
# ==========================================

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Convert model features to numeric and replace invalid values with NaN.
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

text_columns = df.select_dtypes(exclude=np.number).columns
df[text_columns] = df[text_columns].fillna("Unknown")

print("\nAfter preprocessing:")
print("Duplicate Rows:", df.duplicated().sum())
print("Missing Values:", df.isnull().sum().sum())
plt.figure(figsize=(10, 6))

sns.histplot(
    df["vote_average"],
    bins=20,
    kde=True
)

plt.title("Distribution of Movie Ratings")
plt.xlabel("Movie Rating")
plt.ylabel("Number of Movies")

plt.savefig(
    "outputs/eda/rating_distribution.png"
)

plt.show()
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="popularity",
    y="vote_average"
)

plt.title("Popularity vs Movie Rating")
plt.xlabel("Popularity")
plt.ylabel("Movie Rating")

plt.savefig(
    "outputs/eda/popularity_vs_rating.png"
)

plt.show()
# ==========================================
# 5. FEATURE ENGINEERING
# ==========================================

df["release_date"] = pd.to_datetime(
    df["release_date"],
    errors="coerce"
)

df["release_year"] = df[
    "release_date"
].dt.year

df["release_year"] = df[
    "release_year"
].fillna(
    df["release_year"].median()
)

# ==========================================
# 6. SELECT FEATURES AND TARGET
# ==========================================

feature_columns = [
    "budget",
    "popularity",
    "revenue",
    "runtime",
    "vote_count",
    "release_year"
]

model_data = df.dropna(subset=["vote_average"])
X = model_data[feature_columns]
y = model_data["vote_average"]

# ==========================================
# 7. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# ==========================================
# 8. BUILD ML MODELS
# ==========================================

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_prediction = linear_model.predict(X_test)

# Decision Tree
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)
tree_prediction = tree_model.predict(X_test)

# Random Forest
forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
forest_model.fit(X_train, y_train)
forest_prediction = forest_model.predict(X_test)

# Gradient Boosting
gradient_model = GradientBoostingRegressor(random_state=42)
gradient_model.fit(X_train, y_train)
gradient_prediction = gradient_model.predict(X_test)

# ==========================================
# 9. MODEL EVALUATION
# ==========================================

def evaluate_model(
    model_name,
    actual,
    predicted
):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)

    print("\nModel:", model_name)
    print("MAE :", mae)
    print("MSE :", mse)
    print("RMSE:", rmse)
    print("R2  :", r2)

    return mae, mse, rmse, r2


linear_results = evaluate_model(
    "Linear Regression",
    y_test,
    linear_prediction
)

tree_results = evaluate_model(
    "Decision Tree",
    y_test,
    tree_prediction
)

forest_results = evaluate_model(
    "Random Forest",
    y_test,
    forest_prediction
)

gradient_results = evaluate_model(
    "Gradient Boosting",
    y_test,
    gradient_prediction
)

# ==========================================
# 10. MODEL COMPARISON
# ==========================================

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting"
    ],
    "MAE": [
        linear_results[0],
        tree_results[0],
        forest_results[0],
        gradient_results[0]
    ],
    "MSE": [
        linear_results[1],
        tree_results[1],
        forest_results[1],
        gradient_results[1]
    ],
    "RMSE": [
        linear_results[2],
        tree_results[2],
        forest_results[2],
        gradient_results[2]
    ],
    "R2": [
        linear_results[3],
        tree_results[3],
        forest_results[3],
        gradient_results[3]
    ]
})

print("\nMODEL COMPARISON")
print(results)

results.to_csv(
    "outputs/model_results.csv",
    index=False
)

best_model_name = results.loc[
    results["RMSE"].idxmin(),
    "Model"
]

print("\nBest Model:", best_model_name)

# ==========================================
# 11. HYPERPARAMETER TUNING
# ==========================================

parameters = {
    "n_estimators": [
        100,
        200
    ],
    "max_depth": [
        10,
        20,
        None
    ],
    "min_samples_split": [
        2,
        5
    ]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    parameters,
    cv=3,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid_search.best_params_)

# ==========================================
# 12. FINAL MODEL
# ==========================================

best_model = grid_search.best_estimator_
final_prediction = best_model.predict(X_test)

final_results = evaluate_model(
    "Tuned Random Forest",
    y_test,
    final_prediction
)

# ==========================================
# 13. SAVE FINAL MODEL
# ==========================================

joblib.dump(
    best_model,
    "models/movie_rating_model.pkl"
)

print("\nFinal model saved successfully!")
