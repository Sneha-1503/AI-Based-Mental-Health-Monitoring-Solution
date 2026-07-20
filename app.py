import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("mental_health_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(
    page_title="AI Mental Health Monitoring",
    page_icon="🧠",
    layout="wide"
)
st.markdown("""
<style>

.stButton>button{

    background-color:#4CAF50;
    color:white;
    border-radius:10px;
    height:50px;
    font-size:18px;
}

.stButton>button:hover{

    background-color:#45a049;
}

</style>
""", unsafe_allow_html=True)

st.sidebar.title("🧠 AI Mental Health Monitor")

st.sidebar.markdown("---")

st.sidebar.header("Project Information")

st.sidebar.info("""
**Model:** Random Forest

**Accuracy:** 73.78%

**Encoding:** One-Hot Encoding

**Dataset:** Mental Health Survey
""")

st.sidebar.markdown("---")

st.sidebar.success("Developed for Machine Learning Project")


st.markdown("""
# 🧠 AI-Based Mental Health Monitoring System

Predict whether a person is likely to seek mental health treatment based on survey responses.

---
""")

st.write("Fill in the details below to predict whether treatment is recommended.")
gender = st.selectbox(
    "Gender",
    ["Male","Female","Other"]
)

country = st.text_input("Country")

occupation = st.selectbox(
    "Occupation",
    ["Student","Corporate","Business","Housewife","Others"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["Yes","No"]
)

family_history = st.selectbox(
    "Family History",
    ["Yes","No"]
)

days_indoors = st.selectbox(
    "Days Indoors",
    [
        "Go out Every day",
        "1-14 days",
        "15-30 days",
        "More than 2 months"
    ]
)

growing_stress = st.selectbox(
    "Growing Stress",
    ["Yes","No","Maybe"]
)

changes_habits = st.selectbox(
    "Changes Habits",
    ["Yes","No","Maybe"]
)

mental_health_history = st.selectbox(
    "Mental Health History",
    ["Yes","No","Maybe"]
)

mood_swings = st.selectbox(
    "Mood Swings",
    ["High","Medium","Low"]
)

coping_struggles = st.selectbox(
    "Coping Struggles",
    ["Yes","No"]
)

work_interest = st.selectbox(
    "Work Interest",
    ["Yes","No","Maybe"]
)

social_weakness = st.selectbox(
    "Social Weakness",
    ["Yes","No","Maybe"]
)

mental_health_interview = st.selectbox(
    "Mental Health Interview",
    ["Yes","No","Maybe"]
)

care_options = st.selectbox(
    "Care Options",
    ["Yes","No","Not Sure"]
)
#if st.button("Predict"):

predict = st.button(
    "🔍 Predict Mental Health Status",
    use_container_width=True
)

if predict:
    user_data = pd.DataFrame({
        "Gender":[gender],
        "Country":[country],
        "Occupation":[occupation],
        "self_employed":[self_employed],
        "family_history":[family_history],
        "Days_Indoors":[days_indoors],
        "Growing_Stress":[growing_stress],
        "Changes_Habits":[changes_habits],
        "Mental_Health_History":[mental_health_history],
        "Mood_Swings":[mood_swings],
        "Coping_Struggles":[coping_struggles],
        "Work_Interest":[work_interest],
        "Social_Weakness":[social_weakness],
        "mental_health_interview":[mental_health_interview],
        "care_options":[care_options]
    })

    user_data = pd.get_dummies(user_data)

    user_data = user_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    prediction = model.predict(user_data)
    st.write(prediction)


    probability = model.predict_proba(user_data)

    confidence = probability.max()*100

    st.markdown("---")

    st.subheader("Prediction Result")

    if prediction[0] == "Yes":
        st.error("🔴 Treatment Recommended")
    else:
        st.success("🟢 Treatment Not Required")

    st.progress(confidence / 100)

    st.metric(
        label="Confidence",
        value=f"{confidence:.2f}%"
    )

    

    st.markdown("---")
    st.subheader("💡 Recommendations")

    if prediction[0] == "Yes":

        st.warning("""
        - Practice mindfulness and meditation
        - Maintain healthy sleeping habits
        - Exercise regularly
        - Stay connected with family and friends
        - Consult a mental health professional
        """)

    else:

        st.success("""
        - Maintain your healthy lifestyle
        - Continue regular exercise
        - Keep a balanced routine
        - Stay socially connected
        """)

    st.markdown("---")

    st.info(
        "⚠ This application is for educational purposes only and does not replace professional medical advice."
    )
    st.markdown("---")

st.caption(
    "Developed by Sneha Singh | AI-Based Mental Health Monitoring System | Machine Learning Project"
)


