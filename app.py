import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ================================
# 1. Page Configuration & Setup
# ================================
st.set_page_config(page_title="Digital Wellness Recommender", layout="wide")

@st.cache_resource
def load_models():
    reg = joblib.load("best_wellbeing_model.pkl")
    clf = joblib.load("burnout_model.pkl")
    return reg, clf

reg_model, clf_model = load_models()

# ================================
# 2. Sidebar / Inputs
# ================================
st.title("🧠 Digital Wellness & Burnout Recommender")
st.markdown("Enter your daily habits below to analyze your wellbeing and burnout risk.")

with st.sidebar:
    st.header("User Profile")
    age = st.number_input("Age", 18, 100, 22)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    status = st.selectbox("Status", ["Student", "Working Professional", "Both"])
    
    st.header("Daily Habits")
    screen = st.slider("Total Screen Time (hrs)", 0.0, 18.0, 5.0)
    social = st.slider("Social Media (hrs)", 0.0, 12.0, 2.0)
    gaming = st.slider("Gaming (hrs)", 0.0, 10.0, 1.0)
    sleep = st.slider("Sleep Hours", 3.0, 12.0, 7.0)
    exercise = st.slider("Exercise Days/Week", 0, 7, 3)

    st.header("Psychological Metrics")
    anxiety = st.slider("Anxiety Score (0-100)", 0, 100, 30)
    overthinking = st.slider("Overthinking Score (0-100)", 0, 100, 30)
    fatigue = st.slider("Emotional Fatigue (0-100)", 0, 100, 30)

# ================================
# 3. Data Preparation
# ================================
# Creating the DataFrame exactly as the model expects
input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Country": ["Uganda"], # Default or add to sidebar
    "Student_Working_Status": [status],
    "Daily_Social_Media_Hours": [social],
    "Screen_Time_Hours": [screen],
    "Online_Gaming_Hours": [gaming],
    "Daily_Sleep_Hours": [sleep],
    "Content_Type_Preference": ["Entertainment"],
    "Anxiety_Score": [anxiety],
    "Overthinking_Score": [overthinking],
    "Emotional_Fatigue_Score": [fatigue],
    "Exercise_Frequency_per_Week": [exercise]
})

# IMPORTANT: Replicate the Feature Engineering from training
input_data["digital_overload"] = (
    input_data["Daily_Social_Media_Hours"] +
    input_data["Screen_Time_Hours"] +
    input_data["Online_Gaming_Hours"]
)
input_data["sleep_deficit"] = 8 - input_data["Daily_Sleep_Hours"]
input_data["mental_strain"] = (
    input_data["Anxiety_Score"] +
    input_data["Overthinking_Score"] +
    input_data["Emotional_Fatigue_Score"]
)

# ================================
# 4. Predictions & Logic
# ================================
if st.button("Analyze My Wellness", type="primary"):
    
    # Run Predictions
    wellbeing_score = reg_model.predict(input_data)[0]
    burnout_risk = clf_model.predict(input_data)[0]

    # Display Results in Columns
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Wellbeing Index", f"{wellbeing_score:.2f}")
    with col2:
        risk_color = "inverse" if burnout_risk == "High" else "normal"
        st.metric("Burnout Risk Level", str(burnout_risk))

    # Recommendations Logic
    st.divider()
    st.subheader("💡 Personalized Recommendations")
    
    recs = []
    if sleep < 7: recs.append("Aim for at least 7-8 hours of sleep to reduce cognitive load.")
    if screen > 7: recs.append("High screen time detected. Try the 20-20-20 rule (every 20 mins, look 20 feet away for 20 secs).")
    if exercise < 3: recs.append("Physical activity is low. Even a 15-minute walk can boost your wellbeing score.")
    if anxiety > 60: recs.append("Your anxiety score is high. Consider mindfulness or speaking to a counselor.")
    
    for r in recs:
        st.info(r)

    # ================================
    # 5. SHAP Explanations
    # ================================
    st.divider()
    st.subheader("🔍 What influenced your score?")
    
    # Note: Use the preprocessor from the pipeline to transform the single row
    X_processed = reg_model.named_steps["preprocessor"].transform(input_data)
    
    # Use Explainer on the model part of the pipeline
    explainer = shap.Explainer(reg_model.named_steps["model"])
    shap_values = explainer(X_processed)
    
    # Since OneHotEncoding changes feature names, SHAP might show "Feature 0, 1, etc." 
    # unless we map names back, but for now, we'll plot the processed values:
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(plt.gcf())