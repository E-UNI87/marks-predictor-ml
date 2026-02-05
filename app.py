import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
df = pd.read_csv("students.csv")

X = df[["Hours"]]
y = df["Marks"]

model = LinearRegression()
model.fit(X, y)


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

if st.button("🎯 Predict My Marks"):
    prediction = model.predict([[hours]])
    st.success(f"✅ Estimated Marks: {round(prediction[0], 2)} / 100")
    # Prediction for all data (line)
    predicted_all = model.predict(X)

    # Create graph
    fig, ax = plt.subplots()

    ax.scatter(df["Hours"], df["Marks"], label="Actual Marks")
    ax.plot(df["Hours"], predicted_all, label="ML Prediction Line")
    ax.scatter(hours, prediction, marker="*", s=200, label="Your Prediction")

    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Marks")
    ax.set_title("📈 Study Hours vs Marks (ML Model)")
    ax.legend()

    # Show graph in Streamlit
    st.pyplot(fig)
s