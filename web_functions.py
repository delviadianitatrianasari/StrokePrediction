import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st

@st.cache
def load_data():
    # load dataset
    df = pd.read_csv('stroke-dataset.csv')

    X = df[["gender", "age", "hypertension", "heart_disease", "ever_married", "Residence_type", "avg_glucose_level", "bmi"]]
    y = df["stroke"]  

    return df, X, y 

@st.cache
def train_model(X, y):
    model = KNeighborsClassifier(n_neighbors = 3)
    model.fit(X, y)

    score = model.score(X, y)

    return model, score  

def predict(X, y, features):
    model, score = train_model(X, y)

    prediction = model.predict(np.array(features).reshape(1, -1))
    return prediction, score
  
