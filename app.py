import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Page config
st.set_page_config(page_title="CliniPredict", layout="wide")

# Styling
st.markdown("""
<style>
[data-testid="stMetricValue"] {
    font-size: 20px;
}
</style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load("cancer_model.pkl")

# Title
st.title("🧠 CliniPredict-Onco-App")
st.markdown("AI-powered support tool for early cancer risk assessment")
st.markdown("---")

# Sidebar
st.sidebar.header("📋 Patient Info")
patient_name = st.sidebar.text_input("Patient Name")
patient_age = st.sidebar.slider("Age", 10, 100, 30)
patient_gender = st.sidebar.selectbox("Gender", ["Female", "Male"])

st.sidebar.markdown("### 🧍 Patient Summary")
st.sidebar.write(f"""
Name: {patient_name}  
Age: {patient_age}  
Gender: {patient_gender}
""")

st.sidebar.markdown("---")
st.sidebar.info("This tool assists clinicians and does not replace diagnosis.")

# 🔥 MODE SWITCH
mode = st.radio("Select Mode", ["Single Patient", "Bulk Upload"])

# =========================
# 🔬 SINGLE PATIENT MODE
# =========================
if mode == "Single Patient":

    st.header("🔬 Single Patient Analysis")

    col1, col2 = st.columns(2)

    with col1:
        mean_radius = st.slider("Mean Radius", 5.0, 30.0, 14.0)
        mean_texture = st.slider("Mean Texture", 5.0, 40.0, 19.0)
        texture_error = st.slider("Texture Error", 0.0, 5.0, 1.0)

    with col2:
        worst_radius = st.slider("Worst Radius", 10.0, 40.0, 25.0)
        compactness_error = st.slider("Compactness Error", 0.0, 0.5, 0.05)

    colA, colB, colC = st.columns(3)
    colA.metric("Mean Radius", round(mean_radius, 2))
    colB.metric("Worst Radius", round(worst_radius, 2))
    colC.metric("Texture Error", round(texture_error, 2))

    if st.button("🔍 Run Risk Analysis"):

        # Input validation FIRST
        if mean_radius <= 0:
            st.error("Mean radius must be greater than 0")
            st.stop()

        input_data = np.array([[mean_radius, mean_texture, texture_error, worst_radius, compactness_error]])

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]

        risk_score = (1 - probability) if prediction[0] == 0 else probability

        st.subheader("🧾 Clinical Result")

        if prediction[0] == 0:
            st.error("⚠️ HIGH RISK (Malignant)")
        else:
            st.success("✅ LOW RISK (Benign)")

        st.markdown("### 📊 Risk Confidence Level")
        st.progress(int(risk_score * 100))
        st.write(f"Confidence Score: {risk_score:.2f}")

        # Chart
        features = ["Mean Radius","Mean Texture","Texture Error","Worst Radius","Compactness Error"]
        values = [mean_radius, mean_texture, texture_error, worst_radius, compactness_error]

        fig, ax = plt.subplots()
        ax.barh(features, values)
        ax.set_title("Patient Feature Profile")
        st.pyplot(fig)

        st.markdown("### 📌 Model Insight")
        st.write("""
        - Larger tumor size increases risk  
        - Irregular texture is a strong indicator  
        - Shape inconsistency contributes to malignancy prediction  
        """)

        # 🔥 DOWNLOAD REPORT (INSIDE BLOCK!)
        report = f"""
CliniPredict Report

Patient: {patient_name}
Age: {patient_age}
Gender: {patient_gender}

Result: {"High Risk" if prediction[0] == 0 else "Low Risk"}
Confidence: {risk_score:.2f}
"""

        st.download_button(
            "📄 Download Report",
            report,
            file_name="patient_report.txt"
        )

# =========================
# 📂 BULK MODE
# =========================
elif mode == "Bulk Upload":

    st.header("📂 Bulk Patient Upload")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)

        st.subheader("📊 Uploaded Data Preview")
        st.dataframe(df_upload.head())

        try:
            features = [
                "mean radius",
                "mean texture",
                "texture error",
                "worst radius",
                "compactness error"
            ]

            X_upload = df_upload[features]

            predictions = model.predict(X_upload)
            probabilities = model.predict_proba(X_upload)[:, 1]

            df_upload["Prediction"] = predictions
            df_upload["Confidence"] = probabilities

            df_upload["Prediction"] = df_upload["Prediction"].map({
                0: "High Risk",
                1: "Low Risk"
            })

            st.subheader("🧾 Prediction Results")
            st.dataframe(df_upload)

            csv = df_upload.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name="prediction_results.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"⚠️ Error: {e}")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
