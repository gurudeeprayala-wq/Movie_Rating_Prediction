import pandas as pd
import joblib

# Load trained model
model = joblib.load(
    "models/movie_rating_model.pkl"
)

print("Movie Rating Prediction")

# New movie information
new_movie = pd.DataFrame({
    "budget": [10000000],
    "popularity": [50],
    "revenue": [50000000],
    "runtime": [120],
    "vote_count": [500],
    "release_year": [2026]
})

# Make prediction
prediction = model.predict(
    new_movie
)

print(
    "Predicted Movie Rating:",
    prediction[0]
)
