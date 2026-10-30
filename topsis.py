import numpy as np
import pandas as pd


def apply_topsis(data, criteria_weights, criteria_types):
    """Apply the TOPSIS method to rank alternatives based on weighted criteria."""
    df = data.copy()
    criteria = list(criteria_weights.keys())

    if not criteria:
        raise ValueError("criteria_weights cannot be empty.")

    missing = [c for c in criteria if c not in df.columns]
    if missing:
        raise KeyError(f"Missing criteria columns in data: {missing}")

    matrix = df[criteria].astype(float).to_numpy()

    # normalization
    denominator = np.linalg.norm(matrix, axis=0)
    denominator = np.where(denominator == 0, 1e-12, denominator)
    normalized_matrix = matrix / denominator

    weights = np.array([criteria_weights[c] for c in criteria], dtype=float)
    weighted_matrix = normalized_matrix * weights

    ideal_best = []
    ideal_worst = []

    for i, criterion in enumerate(criteria):
        column_values = weighted_matrix[:, i]
        if criteria_types.get(criterion, "benefit") == "benefit":
            ideal_best.append(column_values.max())
            ideal_worst.append(column_values.min())
        else:
            ideal_best.append(column_values.min())
            ideal_worst.append(column_values.max())

    ideal_best = np.array(ideal_best, dtype=float)
    ideal_worst = np.array(ideal_worst, dtype=float)

    distance_best = np.sqrt(np.sum((weighted_matrix - ideal_best) ** 2, axis=1))
    distance_worst = np.sqrt(np.sum((weighted_matrix - ideal_worst) ** 2, axis=1))

    denominator_score = distance_best + distance_worst
    denominator_score = np.where(denominator_score == 0, 1e-12, denominator_score)
    topsis_score = distance_worst / denominator_score

    df["TOPSIS_Score"] = topsis_score
    df["Rank"] = df["TOPSIS_Score"].rank(ascending=False, method="min").astype(int)
    df = df.sort_values(by="Rank").reset_index(drop=True)

    return df
