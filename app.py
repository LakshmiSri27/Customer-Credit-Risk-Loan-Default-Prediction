
import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Credit Risk Predictor",
    page_icon="💳",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

model_package = joblib.load(
    "models/credit_risk_xgboost.pkl"
)

model = model_package["model"]
preprocessor = model_package["preprocessor"]
threshold = model_package["threshold"]


# =========================================================
# HEADER
# =========================================================

st.title("💳 Customer Credit Risk & Loan Default Prediction")

st.markdown(
    """
    **Machine Learning Based Credit Risk Assessment**

    Enter customer and loan information to estimate the
    probability of loan default.
    """
)

st.divider()


# =========================================================
# CUSTOMER INFORMATION
# =========================================================

st.header("👤 Customer Information")

col1, col2, col3 = st.columns(3)


with col1:

    person_age = st.number_input(
        "Age",
        min_value=20,
        max_value=100,
        value=25,
        step=1
    )

    person_income = st.number_input(
        "Annual Income",
        min_value=4000,
        max_value=6000000,
        value=55000,
        step=1000
    )

    person_emp_length = st.number_input(
        "Employment Length (Years)",
        min_value=0.0,
        max_value=50.0,
        value=4.0,
        step=0.5
    )


with col2:

    person_home_ownership = st.selectbox(
        "Home Ownership",
        [
            "RENT",
            "MORTGAGE",
            "OWN",
            "OTHER"
        ]
    )

    cb_person_cred_hist_length = st.number_input(
        "Credit History Length (Years)",
        min_value=2,
        max_value=30,
        value=4,
        step=1
    )

    cb_person_default_on_file = st.selectbox(
        "Previous Default on File",
        [
            "N",
            "Y"
        ]
    )


with col3:

    loan_intent = st.selectbox(
        "Loan Intent",
        [
            "PERSONAL",
            "EDUCATION",
            "MEDICAL",
            "VENTURE",
            "HOMEIMPROVEMENT",
            "DEBTCONSOLIDATION"
        ]
    )

    loan_grade = st.selectbox(
        "Loan Grade",
        [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G"
        ]
    )

    loan_amnt = st.number_input(
        "Loan Amount",
        min_value=500,
        max_value=35000,
        value=10000,
        step=500
    )


# =========================================================
# LOAN INFORMATION
# =========================================================

st.divider()

st.header("🏦 Loan Information")

col4, col5, col6 = st.columns(3)


with col4:

    loan_int_rate = st.number_input(
        "Interest Rate (%)",
        min_value=5.0,
        max_value=25.0,
        value=11.0,
        step=0.01
    )


with col5:

    loan_percent_income = st.number_input(
        "Loan Percent of Income",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.01,
        format="%.2f"
    )


with col6:

    st.info(
        """
        **Loan Percent of Income**

        Represents the proportion of income associated
        with the loan amount.
        """
    )


# =========================================================
# INPUT VALIDATION
# =========================================================

st.divider()

validation_error = False

if person_emp_length > person_age:

    st.error(
        "⚠️ Employment length cannot be greater than age."
    )

    validation_error = True


if cb_person_cred_hist_length > person_age:

    st.warning(
        "⚠️ Credit history length is unusually high compared with age."
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================

predict_button = st.button(
    "🔍 Predict Credit Risk",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button and not validation_error:

    input_data = pd.DataFrame({

        "person_age": [person_age],

        "person_income": [person_income],

        "person_home_ownership": [
            person_home_ownership
        ],

        "person_emp_length": [
            person_emp_length
        ],

        "loan_intent": [
            loan_intent
        ],

        "loan_grade": [
            loan_grade
        ],

        "loan_amnt": [
            loan_amnt
        ],

        "loan_int_rate": [
            loan_int_rate
        ],

        "loan_percent_income": [
            loan_percent_income
        ],

        "cb_person_default_on_file": [
            cb_person_default_on_file
        ],

        "cb_person_cred_hist_length": [
            cb_person_cred_hist_length
        ]
    })


    # -----------------------------------------------------
    # PREPROCESS INPUT
    # -----------------------------------------------------

    processed_input = preprocessor.transform(
        input_data
    )


    # -----------------------------------------------------
    # PREDICT PROBABILITY
    # -----------------------------------------------------

    probability = model.predict_proba(
        processed_input
    )[0][1]


    # -----------------------------------------------------
    # APPLY THRESHOLD
    # -----------------------------------------------------

    prediction = int(
        probability >= threshold
    )


    # =====================================================
    # RESULT
    # =====================================================

    st.divider()

    st.header("📊 Prediction Result")


    result_col1, result_col2 = st.columns(2)


    with result_col1:

        st.metric(
            "Estimated Default Probability",
            f"{probability * 100:.2f}%"
        )


    with result_col2:

        st.metric(
            "Classification Threshold",
            f"{threshold:.2f}"
        )


    # -----------------------------------------------------
    # PROBABILITY BAR
    # -----------------------------------------------------

    st.progress(
        float(probability)
    )


    # -----------------------------------------------------
    # RISK RESULT
    # -----------------------------------------------------

    if prediction == 1:

        st.error(
            "⚠️ Higher Default Risk"
        )

        st.write(
            "The model's estimated default probability "
            "is above the selected classification threshold."
        )

    else:

        st.success(
            "✅ Lower Default Risk"
        )

        st.write(
            "The model's estimated default probability "
            "is below the selected classification threshold."
        )


# =========================================================
# MODEL INFORMATION
# =========================================================

st.divider()

st.header("🤖 Model Information")

model_col1, model_col2, model_col3, model_col4 = st.columns(4)


with model_col1:

    st.metric(
        "Accuracy",
        "93.89%"
    )


with model_col2:

    st.metric(
        "Precision",
        "97.58%"
    )


with model_col3:

    st.metric(
        "Recall",
        "73.91%"
    )


with model_col4:

    st.metric(
        "ROC-AUC",
        "0.9505"
    )


st.caption(
    "Model: XGBoost | Classification threshold: 0.45"
)


# =========================================================
# DISCLAIMER
# =========================================================

st.divider()

st.caption(
    """
    ⚠️ This application is a machine-learning project for
    educational and demonstration purposes. Model predictions
    should not be used as the sole basis for real-world lending
    decisions.
    """
)
