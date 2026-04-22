import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Konfigurasi Halaman
st.set_page_config(page_title="Credit Risk Predictor", layout="wide")

# Load Scaler dan Model
@st.cache_resource
def load_assets():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    models = {
        'LightGBM': pickle.load(open('lgbm_model.pkl', 'rb')),
        'CatBoost': pickle.load(open('catboost_model.pkl', 'rb')),
        'Gradient Boosting': pickle.load(open('gradient_boosting_model.pkl', 'rb'))
    }
    return scaler, models

try:
    scaler, models = load_assets()
except Exception as e:
    st.error(f"Gagal memuat model/scaler: {e}")
    st.stop()

st.title("🚀 Prediksi Risiko Gagal Bayar Pinjaman")
st.markdown("--- ")

# Sidebar untuk Input User
st.sidebar.header("Input Atribut Nasabah")

def user_input_features():
    age = st.sidebar.slider("Umur (Age)", 18, 95, 30)
    income = st.sidebar.number_input("Pendapatan Tahunan (Income)", min_value=0, value=50000)
    home = st.sidebar.selectbox("Status Kepemilikan Rumah (Home)", options=[0, 1, 2, 3], format_func=lambda x: ['RENT', 'OWN', 'MORTGAGE', 'OTHER'][x])
    emp_length = st.sidebar.slider("Lama Bekerja dalam Tahun (Emp_length)", 0, 50, 5)
    intent = st.sidebar.selectbox("Tujuan Pinjaman (Intent)", options=[0, 1, 2, 3, 4, 5], format_func=lambda x: ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'][x])
    amount = st.sidebar.number_input("Jumlah Pinjaman (Amount)", min_value=0, value=10000)
    rate = st.sidebar.slider("Suku Bunga (Rate %)", 0.0, 25.0, 11.0)
    percent_income = st.sidebar.slider("Persentase Pinjaman/Pendapatan", 0.0, 1.0, 0.2)
    default = st.sidebar.selectbox("Pernah Gagal Bayar? (Default)", options=[0, 1], format_func=lambda x: "TIDAK" if x==0 else "YA")
    cred_length = st.sidebar.slider("Panjang Riwayat Kredit (Cred_length)", 0, 30, 5)

    data = {
        'Age': age,
        'Income': income,
        'Home': home,
        'Emp_length': emp_length,
        'Intent': intent,
        'Amount': amount,
        'Rate': rate,
        'Percent_income': percent_income,
        'Default': default,
        'Cred_length': cred_length
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Main Page Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Atribut yang Dimasukkan:")
    st.write(input_df)

    selected_model_name = st.selectbox("Pilih Algoritma Model:", list(models.keys()))
    model = models[selected_model_name]

with col2:
    st.subheader("Hasil Prediksi")
    if st.button("Lakukan Prediksi"):
        # Scaling
        input_scaled = scaler.transform(input_df)
        
        # Prediksi
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        if prediction[0] == 1:
            st.error("⚠️ Nasabah Diprediksi GAGAL BAYAR (DEFAULT)")
        else:
            st.success("✅ Nasabah Diprediksi AMAN (NON-DEFAULT)")
        
        st.write(f"**Probabilitas Gagal Bayar:** {prediction_proba[0][1]:.2%}")
        st.write(f"**Algoritma Digunakan:** {selected_model_name}")
