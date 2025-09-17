import streamlit as st
import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader

# ---------------------------
# Load dataset
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("airlines_reviews.csv")
    df["Recommended"] = (df["Overall Rating"] >= 7).astype(int)
    return df

df = load_data()

# ---------------------------
# Train SVD model (collaborative filtering)
# ---------------------------
@st.cache_resource
def train_model(df):
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(df[["Type of Traveller", "Airline", "Overall Rating"]], reader)
    trainset = data.build_full_trainset()
    svd_model = SVD(random_state=42)
    svd_model.fit(trainset)
    return svd_model

svd_model = train_model(df)

# ---------------------------
# Hybrid Recommendation System
# ---------------------------
def hybrid_recommend(traveller_type, top_n=5):
    summary = df.groupby("Airline").agg(
        Avg_Rating=("Overall Rating", "mean"),
        Recommend_Rate=("Recommended", "mean"),
        Review_Count=("Airline", "count")
    ).reset_index()

    summary["Review_Count_Norm"] = summary["Review_Count"] / summary["Review_Count"].max()

    # Rule-based score
    summary["Rule_Score"] = (
        0.5 * summary["Avg_Rating"]
        + 0.3 * summary["Recommend_Rate"] * 10
        + 0.2 * summary["Review_Count_Norm"] * 10
    )

    # ML predictions (SVD)
    airlines = summary["Airline"].tolist()
    ml_scores = []
    for airline in airlines:
        try:
            pred = svd_model.predict(traveller_type, airline).est
        except:
            pred = summary.loc[summary["Airline"] == airline, "Avg_Rating"].values[0]
        ml_scores.append(pred)

    summary["ML_Score"] = ml_scores
    summary["Hybrid_Score"] = 0.6 * summary["ML_Score"] + 0.4 * summary["Rule_Score"]

    return summary.sort_values(by="Hybrid_Score", ascending=False).head(top_n)

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("✈️ Airline Recommendation System")
st.write("Hybrid recommendation engine combining rule-based + ML (SVD)")

# Inputs
traveller_type = st.selectbox("Select Traveller Type", df["Type of Traveller"].dropna().unique())
top_n = st.slider("Number of recommendations", 1, 10, 5)

# Run recommender
if st.button("Get Recommendations"):
    results = hybrid_recommend(traveller_type, top_n=top_n)
    st.dataframe(results[["Airline", "Avg_Rating", "Rule_Score", "ML_Score", "Hybrid_Score"]])
