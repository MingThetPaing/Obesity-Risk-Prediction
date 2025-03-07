import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# dataset
data = pd.read_csv(r"C:\Users\DELL\Downloads\ObesityDataSet.csv")

# label encoding
categorical_columns = ["Gender", "family_history_with_overweight", "FAVC", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS"]
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

target_column = "NObeyesdad"
if target_column not in data.columns:
    raise KeyError(f"Column '{target_column}' not found in dataset")

# Label encoding target
target_le = LabelEncoder()
data[target_column] = target_le.fit_transform(data[target_column])
label_encoders[target_column] = target_le

# Feature selection
X = data.drop(columns=[target_column])
y = data[target_column]

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Training model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Save model
with open("obesity_model.pkl", "wb") as file:
    pickle.dump((model, scaler, label_encoders, target_le), file)

# Streamlit app
st.title('Obesity Risk Prediction')
st.write(' This is the prediction of obesity levels based on the data of people from the countries of Mexico, Peru and Colombia, with ages between 14 and 61 and diverse eating habits and physical condition ')

# Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=10, max_value=100, value=25)
height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.7)
weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
family_history = st.selectbox("Family History of Overweight", ["yes", "no"])
favc = st.selectbox("Frequent High Caloric Food Consumption", ["yes", "no"])
fcvc = st.slider("Frequency of Vegetable Consumption (1-3)", 1, 3, 2)
ncp = st.slider("Number of Main Meals per Day (1-4)", 1, 4, 3)
caec = st.selectbox("Consumption of Food Between Meals", ["Sometimes", "Frequently", "Always", "no"])
smoke = st.selectbox("Do you smoke?", ["yes", "no"])
ch2o = st.slider("Daily Water Intake (in liters) (1-3)", 1, 3, 2)
scc = st.selectbox("Calories Monitoring", ["yes", "no"])
faf = st.slider("Physical Activity Frequency (0-3)", 0, 1, 2, 3)
calc = st.selectbox("Consumption of Alcohol", ["Sometimes", "Frequently", "Always", "no"])
mtrans = st.selectbox("Mode of Transportation", ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"])

# Recall trained model
with open("obesity_model.pkl", "rb") as file:
    model, scaler, label_encoders, target_le = pickle.load(file)

# Prepare input data
input_data = pd.DataFrame([[gender, age, height, weight, family_history, favc, fcvc, ncp, caec, smoke, ch2o, scc, faf, calc, mtrans]], 
                          columns=X.columns)

# Encode categorical data
for col in categorical_columns:
    input_data[col] = label_encoders[col].transform(input_data[col])

# Scale numerical features
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    result = target_le.inverse_transform(prediction)[0]
    st.success(f"Predicted Obesity Level: {result}")