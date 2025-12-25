import streamlit as st
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Ridge & Lasso Regression", layout="centered")

def load_css(file):
    with open(file) as f:
        st.markdown(f"<style> {f.read()}</style>", unsafe_allow_html=True)
load_css("style_rl.css")

st.title("Ridge & Lasso Regression")
st.write("Predict **Tip Amount** using regularized linear models")

@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

X = df.drop(columns=["tip"])
y = df["tip"]

X = pd.get_dummies(X, drop_first=True)
feature_names = X.columns

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling (important for Ridge & Lasso)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

st.subheader("Choose Regression Model")

model_type = st.radio(
    "Select Model",
    ["Ridge Regression", "Lasso Regression"]
)

alpha = st.slider("Regularization Strength (alpha)", 0.01, 10.0, 1.0)

if model_type == "Ridge Regression":
    model = Ridge(alpha=alpha)
else:
    model = Lasso(alpha=alpha)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance")

c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")

st.metric("R² Score", f"{r2:.3f}")

st.subheader("Actual vs Predicted Tips")

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.6)
ax.plot([y.min(), y.max()], [y.min(), y.max()], color="red")
ax.set_xlabel("Actual Tip")
ax.set_ylabel("Predicted Tip")
st.pyplot(fig)

st.subheader("Model Coefficients")

coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", key=abs, ascending=False)

st.dataframe(coef_df)

st.subheader("Predict Tip Amount")

total_bill = st.slider(
    "Total Bill",
    float(df.total_bill.min()),
    float(df.total_bill.max()),
    30.0
)

size = st.slider("Party Size", 1, 6, 2)
sex = st.selectbox("Sex", ["Female", "Male"])
smoker = st.selectbox("Smoker", ["No", "Yes"])
day = st.selectbox("Day", ["Thur", "Fri", "Sat", "Sun"])
time = st.selectbox("Time", ["Lunch", "Dinner"])

input_data = {
    "total_bill": total_bill,
    "size": size,
    "sex_Male": 1 if sex == "Male" else 0,
    "smoker_Yes": 1 if smoker == "Yes" else 0,
    "day_Fri": 1 if day == "Fri" else 0,
    "day_Sat": 1 if day == "Sat" else 0,
    "day_Sun": 1 if day == "Sun" else 0,
    "day_Thur": 1 if day == "Thur" else 0,
    "time_Dinner": 1 if time == "Dinner" else 0
}

input_df = pd.DataFrame([input_data])

# Align columns
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# Scale & predict
input_scaled = scaler.transform(input_df)
predicted_tip = model.predict(input_scaled)[0]

st.success(f"Predicted Tip Amount: **${predicted_tip:.2f}**")
