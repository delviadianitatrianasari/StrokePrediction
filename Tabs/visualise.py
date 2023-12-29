import streamlit as st
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from web_functions import train_model, load_data 
from streamlit.experimental import suppress_st_warning

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, cmap='coolwarm', xticklabels=['No Stroke', 'Stroke'], yticklabels=['No Stroke', 'Stroke'], fmt='d', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    fig = plt.gcf()  # Simpan referensi ke gambar
    plt.close()  # Tutup plot
    st.pyplot(fig)  # Tampilkan gambar yang disimpan

def knn_visualization(k, X, y_test, y_pred):
    plt.figure(figsize=(10, 8))
    correct_pred = (y_pred == y_test)
    wrong_pred = (y_pred != y_test)
    plt.scatter(X[y_test == 1]['avg_glucose_level'], X[y_test == 1]['age'], color='red', label='Stroke = 1 (Yes)', alpha=0.7, s=50)
    plt.scatter(X[y_test == 0]['avg_glucose_level'], X[y_test == 0]['age'], color='blue', label='Stroke = 0 (No)', alpha=0.7, s=50)
    plt.xlabel('avg_glucose_level')
    plt.ylabel('age')
    plt.title(f'KNN Scatter Plot (K = {k})')
    plt.legend()
    plt.scatter(x=X.iloc[0]['avg_glucose_level'], y=X.iloc[0]['age'], color='yellow', s=300, marker='*')  
    fig = plt.gcf()  # Simpan referensi ke gambar
    plt.close()  # Tutup plot
    st.pyplot(fig)  # Tampilkan gambar yang disimpan
    
def app(df, X, y):
    warnings.filterwarnings('ignore') 
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title("Visualisasi Prediksi Stroke")  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model, score = train_model(X_train, y_train)  # Melatih model menggunakan data latih
    
    if st.checkbox("Plot Confusion Matrix"):
        y_pred = model.predict(X_test)  # Menghasilkan prediksi dari model menggunakan data uji
        plot_confusion_matrix(y_test, y_pred)  # Menampilkan confusion matrix menggunakan data uji
        
    if st.checkbox("Plot K-Neighboors"): 
        k_value = 8  # Ganti sesuai kebutuhan
        knn_visualization(k_value, X_test, y_test, model.predict(X_test))  # Menampilkan plot KNN menggunakan data uji
