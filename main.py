# Description: This App detects if someone has diabetes or not using machine learning algorithms

# Import Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
import streamlit as st

st.write("""
# Diabetes Detector
Detect if someon might have diabetes using machine learning
""")

image = Image.open("image.png")
st.image(image, caption = 'ML', use_column_width=True)

#Data
df = pd.read_csv("diabetes_csv.csv")

st.subheader('Data Information:')
st.dataframe(df)
# Show Statistics

st.write(df.describe())

chart = st.bar_chart(df)

# Split Data into X and Y
X = df.iloc[:, 0:8]
Y = df.iloc[:, -1]

# Spliting data into test and train

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Get feature from input


def get_user_input():
    Pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    GlucoseLevel = st.sidebar.slider('GlucoseLevel', 0, 199, 117)
    BloodPressure = st.sidebar.slider('BloodPressure', 0, 122, 72)
    SkinThickness = st.sidebar.slider('SkinThickness', 0, 99, 23)
    InsulinLevels = st.sidebar.slider('InsulinLevels', 0.0, 846.0, 30.0)
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    PedigreeFunciton = st.sidebar.slider('PedigreeFunction', 0.078, 2.42, 0.3725)
    Age = st.sidebar.slider('Age', 21, 81, 29)


    # Store a dictionary 
    user_data = { 'Pregnancies': Pregnancies,
    'GlucoseLevel':GlucoseLevel,
    'BloodPressure':BloodPressure,
    'SkinThickness':SkinThickness,
    'InsulinLevels':InsulinLevels, 
    'BMI':BMI,
    'PedigreeFunciton ':PedigreeFunciton, 
    'Age':Age }
    # Transform the data into a dataframe
    features = pd.DataFrame(user_data, index=[0])
    return features

#Store the user input into a dataFrame:
user_input = get_user_input()

st.subheader("User Input")
st.write(user_input)

# Create and Train Model

rf = RandomForestClassifier()
rf.fit(X_train, Y_train)

st.subheader("Model Test Accuracy:")

st.write(str(accuracy_score(Y_test, rf.predict(X_test))*100 )+ "%")

# Store Model Predictions

pred = rf.predict(user_input)

#Display
st.subheader('Classification: ')
st.write(pred)
