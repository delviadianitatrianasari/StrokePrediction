import streamlit as st

def app():
    st.markdown(
        """
        <style>
        .css-2trqyj {
            font-family: 'Times New Roman', Times, serif !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("Aplikasi Prediksi Penyakit Stroke")
    st.subheader("Selamat Datang Di Aplikasi Prediksi Penyakit Stroke!")
    st.write("Aplikasi prediksi stroke adalah aplikasi yang digunakan untuk memprediksi risiko seseorang terkena stroke. Aplikasi ini menggunakan berbagai faktor risiko stroke, seperti usia, jenis kelamin, tekanan darah, kolesterol, gula darah, dan gaya hidup, untuk menghitung risiko stroke seseorang.Aplikasi prediksi stroke dapat digunakan untuk membantu orang-orang untuk mengetahui risiko stroke mereka. Dengan mengetahui risiko stroke mereka, orang-orang dapat mengambil tindakan untuk mengurangi risiko stroke mereka, seperti mengonsumsi obat-obatan, mengubah gaya hidup, atau melakukan pemeriksaan kesehatan secara rutin.")
    st.image("Pic1.jpg")
    st.image("Pic2.jpg")
    st.image("Pic3.jpg")