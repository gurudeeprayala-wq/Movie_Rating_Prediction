from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from topsis import apply_topsis


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "movies_metadata.csv"
MODEL_PATH = BASE_DIR / "models" / "movie_rating_model.pkl"
OUTPUT_PATH = BASE_DIR / "outputs" / "movie_ranking.csv"


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    print("Dataset loaded successfully!")

    numeric_columns = [
        "budget",
        "popularity",
        "revenue",
        "runtime",
        "vote_average",
        "vote_count",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.drop_duplicates()

    for column in ["budget", "popularity", "revenue", "runtime", "vote_count"]:
        df[column] = df[column].fillna(df[column].median())

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year
    df["release_year"] = df["release_year"].fillna(df["release_year"].median())

    print("Loading trained model...")
    model = joblib.load(MODEL_PATH)
    print("ML model loaded successfully!")

    feature_columns = [
        "budget",
        "popularity",
        "revenue",
        "runtime",
        "vote_count",
        "release_year",
    ]

    X = df[feature_columns].copy()
    df["Predicted_Rating"] = model.predict(X)
    print("ML predictions generated!")

    ranking_data = df[["title", "Predicted_Rating", "popularity", "vote_count", "revenue"]].copy()
    ranking_data = ranking_data.replace([np.inf, -np.inf], np.nan)
    ranking_data = ranking_data.dropna()

    criteria_weights = {
        "Predicted_Rating": 0.50,
        "popularity": 0.20,
        "vote_count": 0.15,
        "revenue": 0.15,
    }

    criteria_types = {
        "Predicted_Rating": "benefit",
        "popularity": "benefit",
        "vote_count": "benefit",
        "revenue": "benefit",
    }

    result = apply_topsis(ranking_data, criteria_weights, criteria_types)

    print("\n" + "=" * 80)
    print("              TOP MOVIE RANKING")
    print("=" * 80)
    print(
        result[
            ["title", "Predicted_Rating", "popularity", "vote_count", "revenue", "TOPSIS_Score", "Rank"]
        ].head(10).to_string(index=False)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print("\nMovie ranking saved successfully!")
    print(f"File: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
