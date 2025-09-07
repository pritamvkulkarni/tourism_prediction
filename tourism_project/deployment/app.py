import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="pkulkar/tourism-prediction", filename="best_machine_failure_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Wellness Tourism Package Purchase Prediction")
st.write("""
This application predicts the likelihood of a customer purchasing the Wellness Tourism Package based on their details and interaction data.
Please enter the customer information below to get a prediction.
""")

# User input
Age = st.number_input("Age", min_value=0, max_value=150, value=30)
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
CityTier = st.number_input("City Tier", min_value=1, max_value=3, value=1)
Occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Small Business", "Large Business", "Government", "Retired", "Student"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, value=1)
PreferredPropertyStar = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=3)
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
NumberOfTrips = st.number_input("Number of Trips", min_value=0, value=0)
Passport = st.selectbox("Passport", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
OwnCar = st.selectbox("Own Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, value=0)
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP", "Director", "Senior Director"]) # Added sample designations, adjust as needed
MonthlyIncome = st.number_input("Monthly Income", min_value=0.0, value=50000.0)
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Luxury", "Super Deluxe"]) # Added sample products, adjust as needed
NumberOfFollowups = st.number_input("Number of Followups", min_value=0, value=0)
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, value=10.0)


# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'ProductPitched': ProductPitched,
    'NumberOfFollowups': NumberOfFollowups,
    'DurationOfPitch': DurationOfPitch
}])


if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Likely to Purchase" if prediction == 1 else "Not Likely to Purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts the customer is: **{result}**")
