import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st

# Title
st.title("🎯 Student Marks Predictor (ML App)")

# Load data
df = pd.read_csv("students.csv")

X = df[["Hours"]]
y = df["Marks"]

# Train model
model = LinearRegression()
model.fit(X, y)

# User input
hours = st.number_input("Enter study hours:", min_value=0.0, step=0.5)

if st.button("Predict Marks"):
    prediction = model.predict([[hours]])
    st.success(f"Predicted Marks: {round(prediction[0], 2)}")
