import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import os

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="🫀",
    layout="wide"
)

st.title("🫀 Heart Disease Prediction System")
st.markdown("### Predict the likelihood of Heart Disease using Machine Learning")

# -------------------------------------------------
# Check Files
# -------------------------------------------------

if not os.path.exists("heart-disease-model.pkl"):
    st.error("heart-disease-model.pkl not found")
    st.stop()

if not os.path.exists("feature_names.pkl"):
    st.error("feature_names.pkl not found")
    st.stop()

# -------------------------------------------------
# Load Model
# -------------------------------------------------

with open("heart-disease-model.pkl","rb") as file:
    model, scaler = pickle.load(file)

with open("feature_names.pkl","rb") as file:
    feature_names = pickle.load(file)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------

st.sidebar.title("🩺 Patient Information")

age = st.sidebar.number_input(
    "Age",
    min_value=20,
    max_value=100,
    value=45
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male","Female"]
)

cp = st.sidebar.selectbox(
    "Chest Pain Type",
    [0,1,2,3],
    help="0=Typical, 1=Atypical, 2=Non-Anginal, 3=Asymptomatic"
)

bp = st.sidebar.number_input(
    "Resting Blood Pressure",
    min_value=80,
    max_value=200,
    value=120
)

chol = st.sidebar.number_input(
    "Cholesterol",
    min_value=100,
    max_value=600,
    value=200
)

fbs = st.sidebar.selectbox(
    "Fasting Blood Sugar >120",
    [0,1]
)

restecg = st.sidebar.selectbox(
    "Resting ECG",
    [0,1,2]
)

maxhr = st.sidebar.number_input(
    "Maximum Heart Rate",
    min_value=60,
    max_value=220,
    value=150
)

exang = st.sidebar.selectbox(
    "Exercise Induced Angina",
    [0,1]
)

oldpeak = st.sidebar.number_input(
    "ST Depression",
    min_value=0.0,
    max_value=6.0,
    value=1.0,
    step=0.1
)

slope = st.sidebar.selectbox(
    "ST Slope",
    [0,1,2]
)

ca = st.sidebar.selectbox(
    "Major Vessels",
    [0,1,2,3]
)

thal = st.sidebar.selectbox(
    "Thalassemia",
    [1,2,3]
)
# -------------------------------------------------
# Convert Gender
# -------------------------------------------------

gender_value = 1 if gender == "Male" else 0

# -------------------------------------------------
# Create Patient DataFrame
# -------------------------------------------------

patient = pd.DataFrame({
    "Age": [age],
    "Gender": [gender_value],
    "ChestPainType": [cp],
    "RestingBp": [bp],
    "Cholesterol": [chol],
    "FastingBS": [fbs],
    "RestingECG": [restecg],
    "MaxHR": [maxhr],
    "ExerciseAngina": [exang],
    "ST_Depression": [oldpeak],
    "ST_Slope": [slope],
    "MajorVessels": [ca],
    "Thalassemia": [thal]
})

# Save original values for display
patient_display = patient.copy()

# -------------------------------------------------
# Preprocessing
# -------------------------------------------------

categorical_cols = [
    "Gender",
    "ChestPainType",
    "FastingBS",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope",
    "MajorVessels",
    "Thalassemia"
]

numerical_cols = [
    "Age",
    "RestingBp",
    "Cholesterol",
    "MaxHR",
    "ST_Depression"
]

# One-Hot Encoding
patient = pd.get_dummies(
    patient,
    columns=categorical_cols,
    drop_first=True
)

# Match training features
patient = patient.reindex(
    columns=feature_names,
    fill_value=0
)

# Scale numerical features
patient[numerical_cols] = scaler.transform(
    patient[numerical_cols]
)

# -------------------------------------------------
# Prediction Button
# -------------------------------------------------

if st.button("🩺 Predict Heart Disease"):

    prediction = model.predict(patient)[0]

    probability = model.predict_proba(patient)[0]

    risk = probability[1]

    st.markdown("---")

    st.subheader("🩺 Prediction Result")

    if prediction == 1:
        st.error("⚠ High Risk of Heart Disease")
    else:
        st.success("✅ No Heart Disease Detected")
    # -------------------------------------------------
    # Prediction Probability
    # -------------------------------------------------

    st.subheader("📊 Prediction Probability")

    st.write(f"❤️ Heart Disease Probability : **{risk*100:.2f}%**")
    st.write(f"💚 No Heart Disease Probability : **{probability[0]*100:.2f}%**")

    # -------------------------------------------------
    # Risk Meter
    # -------------------------------------------------

    st.subheader("📈 Risk Meter")

    st.progress(float(risk))

    if risk < 0.30:
        st.success("🟢 Low Risk")

    elif risk < 0.70:
        st.warning("🟡 Moderate Risk")

    else:
        st.error("🔴 High Risk")

    # -------------------------------------------------
    # Pie Chart
    # -------------------------------------------------

    chart_data = pd.DataFrame({
        "Prediction": [
            "No Heart Disease",
            "Heart Disease"
        ],
        "Probability": [
            probability[0] * 100,
            probability[1] * 100
        ]
    })

    fig = px.pie(
        chart_data,
        names="Prediction",
        values="Probability",
        title="Prediction Probability"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -------------------------------------------------
    # Patient Details
    # -------------------------------------------------

    st.subheader("📋 Patient Details")

    summary = pd.DataFrame({

        "Feature":[
            "Age",
            "Gender",
            "Chest Pain Type",
            "Resting Blood Pressure",
            "Cholesterol",
            "Fasting Blood Sugar",
            "Resting ECG",
            "Maximum Heart Rate",
            "Exercise Angina",
            "ST Depression",
            "ST Slope",
            "Major Vessels",
            "Thalassemia"
        ],

        "Value":[
            patient_display["Age"][0],
            "Male" if patient_display["Gender"][0] == 1 else "Female",
            patient_display["ChestPainType"][0],
            patient_display["RestingBp"][0],
            patient_display["Cholesterol"][0],
            patient_display["FastingBS"][0],
            patient_display["RestingECG"][0],
            patient_display["MaxHR"][0],
            patient_display["ExerciseAngina"][0],
            patient_display["ST_Depression"][0],
            patient_display["ST_Slope"][0],
            patient_display["MajorVessels"][0],
            patient_display["Thalassemia"][0]
        ]

    })

    st.dataframe(summary, use_container_width=True)
    
        # -------------------------------------------------
    # Health Recommendations
    # -------------------------------------------------

    st.subheader("💡 Health Recommendations")

    if prediction == 1:

        st.error("The prediction indicates a higher risk of heart disease.")

        st.markdown("""
### Recommended Lifestyle Changes

- 🩺 Consult a Cardiologist
- 🥗 Eat a Heart-Healthy Diet
- 🚶 Exercise Regularly (30–45 minutes daily)
- 🚭 Quit Smoking
- 🍺 Avoid Alcohol
- ⚖ Maintain Healthy Weight
- 🩸 Monitor Blood Pressure
- 🧪 Control Cholesterol Levels
- 😴 Get 7–8 Hours of Sleep
- 🧘 Reduce Stress
        """)

    else:

        st.success("No significant signs of heart disease were detected.")

        st.markdown("""
### Maintain a Healthy Lifestyle

- 🥗 Continue Healthy Eating
- 🚶 Exercise Regularly
- 💧 Drink Plenty of Water
- 😴 Sleep 7–8 Hours Daily
- 🩺 Have Regular Health Check-ups
- 🩸 Monitor Blood Pressure Occasionally
- 😊 Manage Stress
        """)

    # -------------------------------------------------
    # Download Report
    # -------------------------------------------------

    report = summary.copy()

    report.loc[len(report)] = [
        "Heart Disease Probability",
        f"{risk*100:.2f}%"
    ]

    report.loc[len(report)] = [
        "Prediction",
        "Heart Disease" if prediction == 1 else "No Heart Disease"
    ]

    csv = report.to_csv(index=False)

    st.download_button(
        label="📥 Download Prediction Report",
        data=csv,
        file_name="Heart_Disease_Report.csv",
        mime="text/csv"
    )

# -------------------------------------------------
# Footer
# -------------------------------------------------

st.markdown("---")

st.markdown(
"""
### ℹ About this App

This Heart Disease Prediction System uses a trained **Logistic Regression Machine Learning model**
to estimate the likelihood of heart disease based on patient health parameters.

⚠ **Disclaimer:** This application is intended for educational purposes only and should not be used as a substitute for professional medical advice or diagnosis.
"""
)

st.markdown("---")

st.caption("Developed by Pavithra Kamath | MSc Data Science & Analytics")
