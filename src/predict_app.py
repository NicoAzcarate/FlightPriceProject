import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Cargar modelo y escalador
model = load_model("models/flight_price_model.keras")
scaler = joblib.load("models/scaler.pkl")

st.title("✈️ Predicción de Precio de Vuelo")

# Inputs del usuario
airline_code = st.number_input("Código de aerolínea (codificado)", min_value=0, step=1)
aircraft_type = st.number_input("Código de tipo de avión", min_value=0, step=1)
departure_hour = st.slider("Hora de salida", 0, 23, 12)
departure_weekday = st.selectbox("Día de la semana (0=Lunes, 6=Domingo)", list(range(7)))
cabin = st.number_input("Código de cabina", min_value=0, step=1)
dep_airport = st.number_input("Aeropuerto de salida (codificado)", min_value=0, step=1)
arr_airport = st.number_input("Aeropuerto de llegada (codificado)", min_value=0, step=1)
is_non_stop = st.radio("¿Es directo?", [0, 1])
days_in_advance = st.slider("Días de antelación", 0, 365, 30)
distance_miles = st.number_input("Distancia (millas)", min_value=0.0)
duration_seconds = st.number_input("Duración total (segundos)", min_value=0.0)

# Preparar datos para predicción
input_data = pd.DataFrame([{
    "airline_code": airline_code,
    "aircraft_type": aircraft_type,
    "departure_hour": departure_hour,
    "departure_weekday": departure_weekday,
    "cabin": cabin,
    "dep_airport": dep_airport,
    "arr_airport": arr_airport,
    "isNonStop": is_non_stop,
    "days_in_advance": days_in_advance,
    "distance_miles": distance_miles,
    "duration_seconds": duration_seconds
}])

# Botón de predicción
if st.button("Predecir precio"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0][0]
    st.success(f"💸 Precio estimado del vuelo: {prediction:.2f} €")


#EJECUTAR EN EL TERMINAL CON:
# streamlit run predict_app.py
