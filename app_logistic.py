import streamlit as st
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Logistic Regression", layout="centered")

def load_css(file):
    with open(file) as f:
        st.markdown(f"<style> {f.read()}</style>", unsafe_allow_html=True)
load_css("style_log.css")

st.title("Logistic Regression")
st.write("Predict whether a customer is a **Smoker** using the Tips Dataset")

@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

X = df.drop(columns=["smoker"])   # Features
y = df["smoker"].map({"No": 0, "Yes": 1})  # Target → Binary

X = pd.get_dummies(X, drop_first=True)

feature_names = X.columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Performance")
st.metric("Accuracy", f"{accuracy:.2f}")

st.text("Classification Report")

st.text(classification_report(y_test, y_pred))

st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.subheader("Predict Smoker Status")

total_bill = st.slider(
    "Total Bill",
    float(df.total_bill.min()),
    float(df.total_bill.max()),
    30.0
)

size = st.slider("Party Size", 1, 6, 2)
sex = st.selectbox("Sex", ["Female", "Male"])
day = st.selectbox("Day", ["Thur", "Fri", "Sat", "Sun"])
time = st.selectbox("Time", ["Lunch", "Dinner"])

# Build input dictionary
input_data = {
    "total_bill": total_bill,
    "size": size,
    "sex_Male": 1 if sex == "Male" else 0,
    "day_Fri": 1 if day == "Fri" else 0,
    "day_Sat": 1 if day == "Sat" else 0,
    "day_Sun": 1 if day == "Sun" else 0,
    "day_Thur": 1 if day == "Thur" else 0,
    "time_Dinner": 1 if time == "Dinner" else 0
}

input_df = pd.DataFrame([input_data])

# Align columns
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# Scale and predict
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

# Output
if prediction == 1:
    st.success(f"Predicted: **Smoker** (Probability: {probability:.2f})")
else:
    st.success(f"Predicted: **Non-Smoker** (Probability: {1-probability:.2f})")
