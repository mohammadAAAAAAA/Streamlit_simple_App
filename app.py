import streamlit as st

import numpy as np
import pandas as pd
import joblib

model, scaler, encoder = joblib.load('model_scaler_encoder.joblib')
st.title('How much mony will you make? :moneybag:')
# ['age', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'hours-per-week', 'native-country']
age = st.number_input("Enter your age:", min_value=0, max_value=120, value=30) 

options = {
    "a private-sector company or enterprise": "Private",
    "work for a local government": "Local-gov",
    "self-employed but not incorporated": "Self-emp-not-inc",
    "work for the federal government": "Federal-gov",
    "work for a state government": "State-gov",
    "self-employed and incorporated": "Self-emp-inc",
    "working without pay": "Without-pay",
    "never worked before": "Never-worked"
}
workclass = st.selectbox("Type of employment:", options.keys())
workclass = options[workclass] # Get the actual value based on the selected option

options = {
    "No formal schooling": "Preschool",
    "Completed grades 1 to 4": "1st-4th",
    "Completed grades 5 to 6": "5th-6th",
    "Completed grades 7 to 8": "7th-8th",
    "Completed grade 9": "9th",
    "Completed grade 10": "10th",
    "Completed grade 11": "11th",
    "Completed grade 12 or high school diploma": "12th",
    "High school graduate": "HS-grad",
    "Attended college but did not complete a degree": "Some-college",
    "Associate's degree in vocational or technical field": "Assoc-voc",
    "Associate's degree in academic field": "Assoc-acdm",
    "Bachelor's degree": "Bachelors",
    "Master's degree": "Masters",
    "Professional school degree (e.g., law, medicine)": "Prof-school",
    "Doctoral degree": "Doctorate"
}
education  = st.selectbox("Education:", options.keys())
education = options[education]

marital_status  = st.selectbox("Marital Status:", ['single' 'married' 'divorced'])

sex = st.select_slider("Choose sex", ['Male','Female'])

options = {
    "Executives and managers": "Exec-managerial",
    "Professional specialties (e.g., engineers, lawyers, doctors)": "Prof-specialty",
    "Technical support occupations": "Tech-support",
    "Sales occupations": "Sales",
    "Administrative and clerical occupations": "Adm-clerical",
    "Craft and repair occupations": "Craft-repair",
    "Transportation and moving occupations": "Transport-moving",
    "Farming, fishing, and forestry occupations": "Farming-fishing",
    "Protective service occupations": "Protective-serv",
    "Machine operators and inspectors": "Machine-op-inspct",
    "private household service workers": "Priv-house-serv",
    "worker involved in handling and cleaning materials or equipment": "Handlers-cleaners",
    "Armed Forces": "Armed-Forces",
    "Other service occupations": "Other-service"
}
occupation  = st.selectbox("Occupation:", options.keys())
occupation = options[occupation]

options = {
    "the child of the household head": "Own-child",
    "the husband of the household head": "Husband",
    "not related to the household head": "Not-in-family",
    "never married, divorced, widowed, or separated": "Unmarried",
    "the wife of the household head": "Wife",
    "related to the household head, but not in one of the specified categories": "Other-relative"
}
relationship  = st.selectbox("Relationship within your household:", options.keys())
relationship = options[relationship]

options = {
    "Black": "Black",
    "White": "White",
    "American Indian or Alaskan Native": "Amer-Indian-Eskimo",
    "Asian or Pacific Islander": "Asian-Pac-Islander",
    "identifying with a race that is not listed specifically": "Other"
}
race  = st.selectbox("Race:", options.keys())
race = options[race]

options = ['United-States', 'Peru', 'Guatemala', 'Mexico', 'Dominican-Republic', 'Ireland', 'Germany', 'Philippines', 
           'Thailand', 'Haiti', 'El-Salvador', 'Puerto-Rico', 'Vietnam', 'South', 'Columbia', 'Japan', 'India', 'Cambodia',
           'Poland', 'Laos', 'England', 'Cuba', 'Taiwan', 'Italy', 'Canada', 'Portugal', 'China', 'Nicaragua', 'Honduras', 
           'Iran', 'Scotland', 'Jamaica', 'Ecuador', 'Yugoslavia', 'Hungary', 'Hong', 'Greece', 'Trinadad&Tobago',
           'US Minor Islands', 'France', 'Holand-Netherlands']
native_country  = st.selectbox("Native Country:", options)

hours_per_week = st.number_input("Hours per week:", min_value=0, max_value=168, value=50) 


def predict(): 
    row = np.array([age, hours_per_week, workclass, education, marital_status, occupation, relationship, race, sex, native_country]) 

    row_encoded = encoder.transform(row[2:])
    row_scaled = scaler.transform(np.hstack((row[:2], row_encoded)))

    prediction = model.predict(row_scaled)
    if prediction[0] == 1: 
        st.success('Your income will be more than 50k :money_with_wings:')
    else: 
        st.error('Your income will be less than 50k :thumbsdown:') 

trigger = st.button('Predict', on_click=predict)

