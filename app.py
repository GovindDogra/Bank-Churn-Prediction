import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.compose import ColumnTransformer
import joblib

# reading pickle models from pickle files
model_churn = joblib.load("model_churn_.pkl")
col_transform = joblib.load("transformer_churn_.pkl")

# title of app
st.title("Customer Churn Prediction for the Bank")

# Collecting input data from user
CreditScore=st.number_input("Credit Score")
Geography=st.selectbox("Geography",["France","Spain","Germany"])
Gender=st.selectbox("Gender",["Male","Female"])
Age=st.number_input("Age", 18,92)
Tenure=st.number_input("Tenure", 0,10)
Balance=st.number_input("Balance")
NumOfProducts=st.number_input("Number of Products", 1,4)
HasCrCard=st.number_input("Has Credit Card", 0,1)
IsActiveMember=st.number_input("Is Active Member", 0,1)
EstimatedSalary=st.number_input("Estimated Salary")

if st.button("Predict"):
    # Creating DataFrame for input
    input_data = pd.DataFrame({
        'CreditScore': [CreditScore],
        'Geography': [Geography],
        'Gender': [Gender],
        'Age': [Age],
        'Tenure': [Tenure],
        'Balance': [Balance],
        'NumOfProducts': [NumOfProducts],
        'HasCrCard': [HasCrCard],
        'IsActiveMember': [IsActiveMember],
        'EstimatedSalary': [EstimatedSalary]
    })

    # Transforming input data
    input_data_transformed = col_transform.transform(input_data)

    # Predicting churn
    y_pred = model_churn.predict(input_data_transformed)

    # Displaying the result
    if y_pred == 1:
        st.error("⚠️ Churn Alert: Customer is likely to leave.")
    else:
        st.success("✅ No Churn: Customer is likely to stay.")


