import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="CliniPredict", layout="wide")

st.title("🧠 CliniPredict-Onco-App")
st.markdown("AI-powered support tool for early cancer risk assessment")

st.markdown("---")

# Load model
model = joblib.load("cancer_model.pkl")

# Sidebar (like real apps 👀)
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

st.markdown("---")

# Prediction
if st.button("🔍 Run Risk Analysis"):

    input_data = np.array([[mean_radius, mean_texture, texture_error, worst_radius, compactness_error]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("🧾 Clinical Result")

    if prediction[0] == 0:
        st.error(f"⚠️ HIGH RISK (Malignant)\nConfidence: {1 - probability:.2f}")
    else:
        st.success(f"✅ LOW RISK (Benign)\nConfidence: {probability:.2f}")

    # Explanation section
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
