import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Movie Rating & Ranking System",
    page_icon="🎬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
    }
    .sub-title {
        font-size: 20px;
        text-align: center;
        color: gray;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">🎬 Movie Rating Prediction & Ranking System</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">Machine Learning + TOPSIS/MCDM + Streamlit</div>',
    unsafe_allow_html=True,
)
st.divider()


@st.cache_data
def load_dataset():
    return pd.read_csv(PROJECT_DIR / "dataset" / "movies_metadata.csv")


@st.cache_resource
def load_model():
    return joblib.load(PROJECT_DIR / "models" / "movie_rating_model.pkl")


try:
    df = load_dataset()
    model = load_model()
except Exception as e:
    st.error(f"Error loading dataset/model: {e}")
    st.stop()

numeric_columns = [
    "budget",
    "popularity",
    "revenue",
    "runtime",
    "vote_average",
    "vote_count",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.drop_duplicates()

for col in ["budget", "popularity", "revenue", "runtime", "vote_count"]:
    df[col] = df[col].fillna(df[col].median())

# release year feature

df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df["release_year"] = df["release_date"].dt.year
df["release_year"] = df["release_year"].fillna(df["release_year"].median())

feature_columns = [
    "budget",
    "popularity",
    "revenue",
    "runtime",
    "vote_count",
    "release_year",
]

X = df[feature_columns]
try:
    df["Predicted_Rating"] = model.predict(X)
except Exception as e:
    st.error(f"Prediction error: {e}")
    st.stop()

st.sidebar.title("⚙️ Ranking Settings")
st.sidebar.subheader("TOPSIS Criteria Weights")
st.sidebar.write("Adjust the importance of each criterion.")

rating_weight = st.sidebar.slider("⭐ Predicted Rating", 0.0, 1.0, 0.50, 0.05)
popularity_weight = st.sidebar.slider("🔥 Popularity", 0.0, 1.0, 0.20, 0.05)
vote_weight = st.sidebar.slider("👥 Vote Count", 0.0, 1.0, 0.15, 0.05)
revenue_weight = st.sidebar.slider("💰 Revenue", 0.0, 1.0, 0.15, 0.05)

total_weight = rating_weight + popularity_weight + vote_weight + revenue_weight
st.sidebar.write(f"**Total Weight: {total_weight:.2f}**")

if abs(total_weight - 1.0) > 0.001:
    st.sidebar.warning("Weights should add up to 1.00")
    st.stop()

ranking_data = df[["title", "Predicted_Rating", "popularity", "vote_count", "revenue"]].copy()
ranking_data = ranking_data.replace([np.inf, -np.inf], np.nan)
ranking_data = ranking_data.dropna()
ranking_data = ranking_data[ranking_data["title"].astype(str).str.strip() != ""]


def apply_topsis(data, weights):
    df_result = data.copy()
    criteria = ["Predicted_Rating", "popularity", "vote_count", "revenue"]
    matrix = df_result[criteria].astype(float).values

    denominator = np.sqrt(np.sum(matrix ** 2, axis=0))
    denominator[denominator == 0] = 1
    normalized_matrix = matrix / denominator

    weight_array = np.array(weights, dtype=float)
    weighted_matrix = normalized_matrix * weight_array

    ideal_best = weighted_matrix.max(axis=0)
    ideal_worst = weighted_matrix.min(axis=0)

    distance_best = np.sqrt(np.sum((weighted_matrix - ideal_best) ** 2, axis=1))
    distance_worst = np.sqrt(np.sum((weighted_matrix - ideal_worst) ** 2, axis=1))

    denominator_score = distance_best + distance_worst
    denominator_score[denominator_score == 0] = 1
    topsis_score = distance_worst / denominator_score

    df_result["TOPSIS_Score"] = topsis_score
    df_result["Rank"] = df_result["TOPSIS_Score"].rank(ascending=False, method="min").astype(int)
    df_result = df_result.sort_values("Rank").reset_index(drop=True)
    return df_result

weights = [rating_weight, popularity_weight, vote_weight, revenue_weight]
ranking_result = apply_topsis(ranking_data, weights)

st.subheader("📊 System Overview")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🎬 Movies", len(ranking_result))
with col2:
    st.metric("⭐ Avg Predicted Rating", f"{ranking_result['Predicted_Rating'].mean():.2f}")
with col3:
    st.metric("🏆 Top TOPSIS Score", f"{ranking_result['TOPSIS_Score'].max():.3f}")
with col4:
    st.metric("🥇 Top Rank", "1")

st.divider()

st.subheader("🏆 Top Ranked Movie")
top_movie = ranking_result.iloc[0]
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Movie", str(top_movie["title"]))
with col2:
    st.metric("Predicted Rating", f"{top_movie['Predicted_Rating']:.2f}")
with col3:
    st.metric("TOPSIS Score", f"{top_movie['TOPSIS_Score']:.3f}")
with col4:
    st.metric("Rank", int(top_movie["Rank"]))

st.divider()

st.subheader("🏆 Top 10 Movie Ranking")
top_10 = ranking_result.head(10).copy()
top_10_display = top_10[["Rank", "title", "Predicted_Rating", "popularity", "vote_count", "revenue", "TOPSIS_Score"]].copy()
top_10_display.columns = ["Rank", "Movie", "Predicted Rating", "Popularity", "Vote Count", "Revenue", "TOPSIS Score"]
st.dataframe(top_10_display, use_container_width=True, hide_index=True)

st.divider()

st.subheader("🔎 Search Movie")
search_text = st.text_input("Enter movie name")
if search_text:
    search_result = ranking_result[ranking_result["title"].astype(str).str.contains(search_text, case=False, na=False)]
    if len(search_result) == 0:
        st.warning("No movie found.")
    else:
        st.dataframe(
            search_result[["Rank", "title", "Predicted_Rating", "TOPSIS_Score", "popularity", "vote_count", "revenue"]],
            use_container_width=True,
            hide_index=True,
        )

st.divider()

st.subheader("⚖️ Current TOPSIS Weights")
weights_df = pd.DataFrame({
    "Criterion": ["Predicted Rating", "Popularity", "Vote Count", "Revenue"],
    "Weight": [rating_weight, popularity_weight, vote_weight, revenue_weight],
})
weights_df["Weight (%)"] = weights_df["Weight"] * 100
st.dataframe(weights_df[["Criterion", "Weight (%)"]], use_container_width=True, hide_index=True)

st.subheader("📈 Top 10 TOPSIS Scores")
chart_data = top_10.set_index("title")["TOPSIS_Score"]
st.bar_chart(chart_data)

st.subheader("📥 Download Ranking")
csv_data = ranking_result.to_csv(index=False).encode("utf-8")
st.download_button("Download Movie Ranking CSV", csv_data, file_name="movie_ranking.csv", mime="text/csv")

st.divider()
st.markdown(
    """
    ### 🎬 Movie Rating Prediction & MCDM System

    **Machine Learning → Predicted Rating → TOPSIS/MCDM → Movie Ranking → Streamlit**

    Machine Learning predicts the movie rating, while TOPSIS combines multiple criteria to generate the final movie ranking.
    """
)
