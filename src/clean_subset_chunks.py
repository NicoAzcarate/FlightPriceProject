import pandas as pd
import numpy as np
import re
import os

# Columnas del itinerario (sin datos de segmentos)
usecols = [
    'legId',
    'searchDate', 'flightDate',
    'startingAirport', 'destinationAirport',
    'isNonStop', 'isBasicEconomy', 'isRefundable',
    'baseFare', 'totalFare', 'seatsRemaining',
    'fareBasisCode', 'totalTravelDistance', 'elapsedDays'
]

def process_itinerary_chunk(chunk):
    chunk['searchDate'] = pd.to_datetime(chunk['searchDate'], errors='coerce')
    chunk['flightDate'] = pd.to_datetime(chunk['flightDate'], errors='coerce')
    chunk = chunk.dropna(subset=['searchDate', 'flightDate'])

    # Días de antelación
    chunk['days_in_advance'] = (chunk['flightDate'] - chunk['searchDate']).dt.days
    chunk = chunk[chunk['days_in_advance'] >= 0]

    # Codificar variables categóricas básicas
    for col in ['startingAirport', 'destinationAirport', 'fareBasisCode']:
        chunk[col] = chunk[col].astype('category').cat.codes

    for col in ['isBasicEconomy', 'isRefundable', 'isNonStop']:
        chunk[col] = chunk[col].astype(bool).astype(int)

    # Filtrar precios extremos
    chunk = chunk.dropna(subset=['totalFare'])
    chunk = chunk[chunk['totalFare'] < 5000]

    # Columnas finales
    final_cols = [
        'legId',
        'startingAirport', 'destinationAirport', 'fareBasisCode',
        'isBasicEconomy', 'isRefundable', 'isNonStop',
        'baseFare', 'totalFare', 'seatsRemaining', 'elapsedDays',
        'days_in_advance', 'totalTravelDistance'
    ]

    return chunk[final_cols]

def run_clean_subset_pipeline(subset_path):
    os.makedirs('../cleaned_chunks', exist_ok=True)

    chunk_size = 1_000_000  # Más pequeño para el subset

    for i, chunk in enumerate(pd.read_csv(subset_path, usecols=usecols, chunksize=chunk_size)):
        print(f"🔄 Procesando chunk {i} del subset...")
        cleaned = process_itinerary_chunk(chunk)
        cleaned.to_parquet(f'../cleaned_chunks/chunk_{i}.parquet', index=False)
        print(f"✅ Guardado chunk {i} con {len(cleaned)} filas")
