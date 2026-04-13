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

# Custom CSS for a better UI
st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

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
st.markdown("Analyze your habits and get AI-driven insights into your mental wellbeing.")

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
input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Country": ["Uganda"],
    "Student_Working_Status": [status],
    "Daily_Social_Media_Hours": [social],
    "Screen_Time_Hours": [screen],
    "Online_Gaming_Hours": [gaming],
    "Daily_Sleep_Hours": [sleep],
    "Content_Type_Preference": ["Entertainment"],
    "Anxiety_Score": [anxiety],
    "Overthinking_Score": [overthinking],
    "Emotional_Fatigue_Score": [fatigue],
    "Exercise_Frequency_per_Week": [exercise],

    # Filling missing columns expected by the model
    "Night_Scrolling_Frequency": [3],
    "Social_Comparison_Index": [3],
    "Caffeine_Intake_Cups": [1],
    "Sleep_Quality_Score": [5],
    "Mood_Stability_Score": [5],
    "Motivation_Level": [5],
    "Study_Work_Hours_per_Day": [8]
})

# Feature Engineering
input_data["digital_overload"] = input_data["Daily_Social_Media_Hours"] + input_data["Screen_Time_Hours"] + input_data["Online_Gaming_Hours"]
input_data["sleep_deficit"] = 8 - input_data["Daily_Sleep_Hours"]
input_data["mental_strain"] = input_data["Anxiety_Score"] + input_data["Overthinking_Score"] + input_data["Emotional_Fatigue_Score"]

# ================================
# 4. Predictions & Results
# ================================
if st.button("Analyze My Wellness", type="primary"):
    
    # Run Predictions
    wellbeing_score = reg_model.predict(input_data)[0]
    burnout_raw = clf_model.predict(input_data)[0]

    # Map Burnout Integer to Label
    # Adjust this dictionary to match your model's specific classes
    risk_map = {0: "Low", 1: "Medium", 2: "High"}
    burnout_label = risk_map.get(burnout_raw, "Moderate")

    # Display Metrics
    st.subheader("📊 Your Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Wellbeing Index", f"{wellbeing_score:.2f}")
    with col2:
        st.metric("Burnout Risk Level", burnout_label)

    # --- RECOMMENDATIONS ---
    st.divider()
    st.subheader("💡 Personalized Recommendations")
    
    recs = []
    # Use .iloc[0] to access the values correctly for conditions
    if input_data["Daily_Sleep_Hours"].iloc[0] < 7: 
        recs.append("😴 **Sleep Priority:** Your sleep is below 7 hours. Try setting a 'no-screens' alarm 30 mins before bed.")
    if input_data["Screen_Time_Hours"].iloc[0] > 7: 
        recs.append("📱 **Digital Detox:** High screen time detected. Apply the 20-20-20 rule to reduce eye strain.")
    if input_data["Exercise_Frequency_per_Week"].iloc[0] < 3: 
        recs.append("🏃 **Movement:** Physical activity is lower than recommended. A quick 15-minute walk can boost your mood.")
    if input_data["Anxiety_Score"].iloc[0] > 60: 
        recs.append("🧘 **Stress Management:** High anxiety score. Consider short daily breathing exercises or meditation.")

    if not recs:
        st.success("✅ You're maintaining healthy habits! Keep it up.")
    else:
        for r in recs:
            st.info(r)

    # ================================
    # 5. SHAP Beeswarm Plot
    # ================================
    st.divider()
    st.subheader("🔍 Feature Importance (Beeswarm)")
    
    with st.spinner("Calculating feature impact..."):
        # Get feature names from the preprocessor
        preprocessor = reg_model.named_steps["preprocessor"]
        feature_names = preprocessor.get_feature_names_out()
        
        # Transform data
        X_processed = preprocessor.transform(input_data)
        
        # Get SHAP values
        explainer = shap.Explainer(reg_model.named_steps["model"])
        shap_values = explainer(X_processed)
        
        # Manually assign feature names to the SHAP object
        shap_values.feature_names = list(feature_names)
        
        # Plotting Beeswarm
        # Note: With only 1 row, beeswarm looks like a bar chart but correctly labeled.
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.beeswarm(shap_values, show=False)
        plt.tight_layout()
        st.pyplot(fig)