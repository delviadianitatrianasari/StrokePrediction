import numpy as np
from web_functions import predict
from web_functions import load_data
import streamlit as st


def app(df, X, y):
    st.title("Prediksi Penyakit Stroke")
    st.image('Stroke.png')
    st.write('Enter the following features to stroke predict :')

    gender_dict = {'Male': 1, 'Female': 0}
    hypertension_dict = {'Yes': 1, 'No': 0}
    heart_disease_dict = {'Yes': 1, 'No': 0}
    ever_married_dict = {'Yes': 1, 'No': 0}
    residence_type_dict = {'Rural': 0, 'Urban': 1}

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox('Jenis Kelamin', ('Male', 'Female'))
    with col2:
        age = st.number_input('Usia Pasien', min_value=0, max_value=150, value=30)
    with col1:
        hypertension = st.selectbox('Tekanan Darah Tinggi', ('No', 'Yes'))
    with col2:
        heart_disease = st.selectbox('Penyakit Jantung', ('No', 'Yes'))
    with col1:
        ever_married = st.selectbox('Sudah Menikah?', ('No', 'Yes'))
    with col2:
        residence_type = st.selectbox('Tipe Tempat Tinggal', ('Rural', 'Urban'))
    with col1:
        avg_glucose_level = st.number_input('Rata-rata Kadar Glukosa dalam Darah')
    with col2:  
        bmi = st.number_input('Indeks Massa Tubuh (BMI)')

    prediction = None 

    if st.button('Stroke Prediction'):
        df, X, y = load_data()
        gender_val = gender_dict[gender]
        hypertension_val = hypertension_dict[hypertension]
        heart_disease_val = heart_disease_dict[heart_disease]
        ever_married_val = ever_married_dict[ever_married]
        residence_type_val = residence_type_dict[residence_type]

        st.info("Prediksi Sukses...")

    
        features = [gender_val, age, hypertension_val, heart_disease_val, ever_married_val, residence_type_val, avg_glucose_level, bmi]

        prediction, score = predict(X, y, features)
    

    if prediction is not None:
        if prediction == 1:
            stroke_prediction_text = 'PASIEN TERKENA STROKE'
        else:
            stroke_prediction_text = 'PASIEN TIDAK TERKENA STROKE'
        st.success(stroke_prediction_text)

        st.write("Tingkat Akurasi Model Yang Digunakan",(score*100),"%")