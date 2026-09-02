import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Student Risk Prediction System",
    layout="wide"
)

MODEL_PATH = "student_risk_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def generate_intervention(row):
    recommendations = []

    if "absences" in row.index and pd.notna(row["absences"]) and row["absences"] > 10:
        recommendations.append("Attendance counselling")

    if "failures" in row.index and pd.notna(row["failures"]) and row["failures"] >= 1:
        recommendations.append("Remedial classes")

    if "studytime" in row.index and pd.notna(row["studytime"]) and row["studytime"] <= 1:
        recommendations.append("Study plan support")

    if "goout" in row.index and pd.notna(row["goout"]) and row["goout"] >= 4:
        recommendations.append("Reduce distractions / mentoring")

    if "famsup" in row.index and str(row["famsup"]).strip().lower() == "no":
        recommendations.append("Family support follow-up")

    if "schoolsup" in row.index and str(row["schoolsup"]).strip().lower() == "no":
        recommendations.append("Offer school support")

    if "higher" in row.index and str(row["higher"]).strip().lower() == "no":
        recommendations.append("Motivation counselling for higher studies")

    if not recommendations:
        recommendations.append("Continue monitoring")

    return recommendations


def prepare_input(df, model):
    df = df.copy()

    for col in ["risk", "G3"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    expected_cols = list(model.named_steps["preprocessor"].feature_names_in_)

    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        return None, missing_cols

    df = df[expected_cols]
    return df, []


def main():
    st.title("AI-Powered Student Risk Prediction System")
    st.write(
        "Upload a student CSV file to predict failure risk and generate intervention suggestions."
    )

    model = load_model()

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is None:
        st.info("Upload a CSV file to begin.")
        return

    try:
        df = pd.read_csv(uploaded_file, sep=";")
    except Exception as e:
        st.error(f"Could not read CSV file: {e}")
        return

    input_df, missing_cols = prepare_input(df, model)

    if missing_cols:
        st.error("Missing required columns in uploaded file:")
        st.write(missing_cols)
        return

    try:
        probabilities = model.predict_proba(input_df)[:, 1]
        predictions = model.predict(input_df)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return

    results = input_df.copy()
    results.insert(
    0,
    "Student_ID",
    range(1, len(results) + 1)
    )
    results["Risk_Probability"] = probabilities
    results["Predicted_Risk"] = predictions
    results["Risk_Label"] = np.where(results["Predicted_Risk"] == 1, "High Risk", "Safe")
    results["Interventions"] = results.apply(generate_intervention, axis=1)

    st.subheader("Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Students", len(results))
    col2.metric("High Risk", int((results["Predicted_Risk"] == 1).sum()))
    col3.metric("Safe", int((results["Predicted_Risk"] == 0).sum()))

    st.subheader("Prediction Table")
    st.dataframe(
        results[["Risk_Probability", "Predicted_Risk", "Risk_Label", "Interventions"]],
        use_container_width=True
    )

    st.download_button(
        label="Download Predictions as CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="student_risk_predictions.csv",
        mime="text/csv"
    )

    st.subheader("High-Risk Students")

    high_risk = results[results["Predicted_Risk"] == 1]

    if len(high_risk) > 0:
        st.dataframe(
            high_risk[
                [
                    "Student_ID",
                    "Risk_Probability",
                    "Risk_Label",
                    "Interventions"
                ]
            ].reset_index(drop=True),
            use_container_width=True
        )
    else:
        st.success("No students were flagged as high risk.")


if __name__ == "__main__":
    main()