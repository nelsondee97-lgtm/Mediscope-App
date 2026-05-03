import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="CliniPredict", layout="centered")

st.title("🧠 CliniPredict-Onco-App")
st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("### AI-powered early risk detection based on tumor characteristics")

# Load model
model = joblib.load("cancer_model.pkl")

st.markdown("## 📝 Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    mean_radius = st.number_input("Mean Radius", 0.0, 50.0, 10.0)
    mean_texture = st.number_input("Mean Texture", 0.0, 50.0, 15.0)

with col2:
    texture_error = st.number_input("Texture Error", 0.0, 10.0, 1.0)
    worst_radius = st.number_input("Worst Radius", 0.0, 50.0, 20.0)
    compactness_error = st.number_input("Compactness Error", 0.0, 1.0, 0.05)

st.markdown("---")

if st.button("🔍 Predict Risk"):
    input_data = np.array([[mean_radius, mean_texture, texture_error, worst_radius, compactness_error]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("🧾 Prediction Result")

    if prediction[0] == 0:
        st.error(f"⚠️ High Risk (Malignant)\n\nConfidence: {1 - probability:.2f}")
    else:
        st.success(f"✅ Low Risk (Benign)\n\nConfidence: {probability:.2f}")

    st.markdown("---")
    st.markdown("### 💡 Insight")
    st.info("Tumor size and texture irregularity are key indicators in cancer detection.")
