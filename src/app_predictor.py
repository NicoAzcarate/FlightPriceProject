import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import json

# ------------------------------
# CARGA DE MODELO Y SCALER
# ------------------------------
from pathlib import Path
import tensorflow as tf

# Ruta absoluta desde el archivo actual
BASE_DIR = Path(__file__).resolve().parent.parent  # esto sube desde /src a la raíz del proyecto
MODEL_PATH = BASE_DIR / "models" / "flight_price_predictor_v4_clean_final.h5"

# Cargar el modelo
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(BASE_DIR / "models" / "standard_scaler.pkl")


# ------------------------------
# CARGA DE VALORES POSIBLES
# ------------------------------
import json
from pathlib import Path

# Ruta absoluta al archivo JSON
BASE_DIR = Path(__file__).resolve().parent.parent
COLUMN_VALUES_PATH = BASE_DIR / "data_subsets" / "column_values.json"

# Cargar el JSON
with open(COLUMN_VALUES_PATH, "r") as f:
    column_values = json.load(f)

# Columnas de entrada en orden (según entrenaste el modelo)
feature_order = [
    'segment_num', 'airline_code', 'aircraft_type', 'duration_seconds', 'distance_miles',
    'departure_hour', 'departure_weekday', 'cabin', 'arr_airport', 'dep_airport',
    'startingAirport', 'destinationAirport', 'fareBasisCode', 'isBasicEconomy',
    'isRefundable', 'isNonStop', 'baseFare', 'seatsRemaining', 'elapsedDays',
    'days_in_advance', 'totalTravelDistance'
]

# ------------------------------
# INTERFAZ
# ------------------------------
st.set_page_config(page_title="Predicción de Precios de Vuelos", layout="centered")
st.title("✈️ Predicción de precios de vuelos")
st.markdown("Selecciona los valores para predecir el precio del vuelo:")

inputs = {}

for col in feature_order:
    if col in column_values:
        options = sorted([int(v) if isinstance(v, str) and v.isdigit() else v for v in column_values[col]])

        inputs[col] = st.selectbox(f"{col}", options)
    elif col in ['duration_seconds', 'distance_miles', 'departure_hour', 'departure_weekday',
                 'baseFare', 'seatsRemaining', 'elapsedDays', 'days_in_advance', 'totalTravelDistance']:
        inputs[col] = st.number_input(f"{col}", min_value=0.0, step=1.0)
    else:
        inputs[col] = st.number_input(f"{col}", step=1.0)

# ------------------------------
# PREDICCIÓN
# ------------------------------
if st.button("📊 Predecir precio"):
    input_array = np.array([[inputs[col] for col in feature_order]])
    input_scaled = scaler.transform(input_array)
    pred_scaled = model.predict(input_scaled)
    pred_price = float(pred_scaled[0][0]) * 1000

    st.success(f"💶 Precio estimado: {pred_price:.2f} €")