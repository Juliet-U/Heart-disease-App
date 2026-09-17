from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "rf_pipeline.pkl"

NUMERICAL_FEATURES = [
    "age",
    "resting_blood_pressure",
    "cholesterol",
    "max_heart_rate",
    "st_depression",
]

FEATURES = [
    "age",
    "sex",
    "chest_pain_type",
    "resting_blood_pressure",
    "cholesterol",
    "fasting_blood_sugar",
    "ecg",
    "max_heart_rate",
    "exercise_induced_chest_pain",
    "st_depression",
    "st_slope",
    "stained_blood_vessels",
    "blood_disorder",
]


@st.cache_resource
def load_pipeline():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file was not found at {MODEL_PATH}. "
            "Place rf_pipeline.pkl in the same folder as app.py."
        )
    return joblib.load(MODEL_PATH)


def main():
    st.set_page_config(
        page_title="Heart Disease Predictor",
        page_icon=":heart:",
        layout="centered",
    )

    st.title("Heart Disease Predictor")
    st.write("Enter patient information to estimate the model's predicted probability.")

    try:
        pipeline = load_pipeline()
    except Exception as error:
        st.error(f"Unable to load the model: {error}")
        st.stop()

    model = pipeline["model"]
    scaler = pipeline["scaler"]
    trained_features = pipeline.get("features", FEATURES)

    if list(trained_features) != FEATURES:
        st.error(
            "The saved model feature order does not match this app. "
            f"Expected: {FEATURES}; found: {list(trained_features)}"
        )
        st.stop()

    with st.form("patient_form"):
        st.subheader("Patient information")

        left_column, right_column = st.columns(2)
        with left_column:
            age = st.number_input("Age", min_value=1, max_value=120, value=54, step=1)
            sex = st.selectbox("Sex", options=[0, 1], format_func=lambda value: "Female" if value == 0 else "Male")
            chest_pain_type = st.selectbox("Chest pain type", options=[0, 1, 2, 3])
            resting_blood_pressure = st.number_input(
                "Resting blood pressure", min_value=50, max_value=250, value=130, step=1
            )
            cholesterol = st.number_input(
                "Cholesterol", min_value=0, max_value=700, value=246, step=1
            )
            fasting_blood_sugar = st.selectbox(
                "Fasting blood sugar > 120 mg/dL", options=[0, 1], format_func=lambda value: "No" if value == 0 else "Yes"
            )
            ecg = st.selectbox("Resting ECG", options=[0, 1, 2])

        with right_column:
            max_heart_rate = st.number_input(
                "Maximum heart rate", min_value=40, max_value=250, value=150, step=1
            )
            exercise_induced_chest_pain = st.selectbox(
                "Exercise-induced chest pain", options=[0, 1], format_func=lambda value: "No" if value == 0 else "Yes"
            )
            st_depression = st.number_input(
                "ST depression", min_value=0.0, max_value=10.0, value=1.2, step=0.1, format="%.1f"
            )
            st_slope = st.selectbox("ST slope", options=[0, 1, 2])
            stained_blood_vessels = st.selectbox("Number of stained blood vessels", options=[0, 1, 2, 3])
            blood_disorder = st.selectbox("Blood disorder", options=[0, 1, 2, 3])

        submitted = st.form_submit_button("Predict probability", type="primary")

    if submitted:
        patient_data = pd.DataFrame(
            [[
                age,
                sex,
                chest_pain_type,
                resting_blood_pressure,
                cholesterol,
                fasting_blood_sugar,
                ecg,
                max_heart_rate,
                exercise_induced_chest_pain,
                st_depression,
                st_slope,
                stained_blood_vessels,
                blood_disorder,
            ]],
            columns=FEATURES,
        )

        patient_data[NUMERICAL_FEATURES] = scaler.transform(
            patient_data[NUMERICAL_FEATURES]
        )

        probabilities = model.predict_proba(patient_data)[0]
        no_disease_probability = probabilities[0]
        disease_probability = probabilities[1]

        st.subheader("Prediction")
        result_column, probability_column = st.columns(2)
        with result_column:
            if disease_probability >= 0.5:
                st.error("Higher predicted probability of heart disease")
            else:
                st.success("Lower predicted probability of heart disease")
        with probability_column:
            st.metric("Heart disease probability", f"{disease_probability:.1%}")

        st.progress(float(disease_probability))
        st.caption(f"No heart disease probability: {no_disease_probability:.1%}")
        st.warning(
            "This tool is for educational purposes only and is not a medical diagnosis. "
            "Consult a qualified healthcare professional for medical advice."
        )


if __name__ == "__main__":
    main()
