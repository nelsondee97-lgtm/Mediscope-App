import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Page config FIRST
st.set_page_config(page_title="CliniPredict", layout="wide")

# Styling
st.markdown("""
<style>
[data-testid="stMetricValue"] {
    font-size: 20px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🧠 CliniPredict-Onco-App")
st.markdown("AI-powered support tool for early cancer risk assessment")
st.markdown("---")

# Load model
model = joblib.load("cancer-model.pkl")

# Sidebar
st.sidebar.header("📋 Patient Info")
patient_name = st.sidebar.text_input("Patient Name")
patient_age = st.sidebar.slider("Age", 10, 100, 30)
patient_gender = st.sidebar.selectbox("Gender", ["Female", "Male"])

st.sidebar.markdown("---")
st.sidebar.info("This tool assists clinicians and does not replace diagnosis.")

# Main layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔬 Tumor Measurements")
    mean_radius = st.slider("Mean Radius", 5.0, 30.0, 14.0)
    mean_texture = st.slider("Mean Texture", 5.0, 40.0, 19.0)
    texture_error = st.slider("Texture Error", 0.0, 5.0, 1.0)

with col2:
    st.subheader("📊 Advanced Metrics")
    worst_radius = st.slider("Worst Radius", 10.0, 40.0, 25.0)
    compactness_error = st.slider("Compactness Error", 0.0, 0.5, 0.05)

# 🔥 Metrics (AFTER inputs!)
colA, colB, colC = st.columns(3)
colA.metric("Mean Radius", round(mean_radius, 2))
colB.metric("Worst Radius", round(worst_radius, 2))
colC.metric("Texture Error", round(texture_error, 2))

st.markdown("---")

# Prediction button
if st.button("🔍 Run Risk Analysis"):

    input_data = np.array([[mean_radius, mean_texture, texture_error, worst_radius, compactness_error]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    risk_score = (1 - probability) if prediction[0] == 0 else probability

    st.subheader("🧾 Clinical Result")

    if prediction[0] == 0:
        st.error("⚠️ HIGH RISK (Malignant)")
    else:
        st.success("✅ LOW RISK (Benign)")

    # 🔥 Progress bar
    st.markdown("### 📊 Risk Confidence Level")
    st.progress(int(risk_score * 100))
    st.write(f"Confidence Score: {risk_score:.2f}")

    # 🔥 Feature chart
    st.markdown("### 🧠 Patient Feature Profile")

    features = [
        "Mean Radius",
        "Mean Texture",
        "Texture Error",
        "Worst Radius",
        "Compactness Error"
    ]

    values = [mean_radius, mean_texture, texture_error, worst_radius, compactness_error]

    fig, ax = plt.subplots()
    ax.barh(features, values)
    ax.set_title("Patient Feature Profile")

    st.pyplot(fig)

    # Explanation
    st.markdown("### 📌 Model Insight")
    st.write("""
    - Larger tumor size increases risk  
    - Irregular texture is a strong indicator  
    - Shape inconsistency contributes to malignancy prediction  
    """)

    # Patient summary
    st.markdown("### 🧍 Patient Summary")
    st.write(f"""
    Name: {patient_name}  
    Age: {patient_age}  
    Gender: {patient_gender}
    """)
