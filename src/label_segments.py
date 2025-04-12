import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import joblib

# Columnas necesarias para procesar segmentos
usecols = [
    'legId', 'segmentsAirlineCode', 'segmentsEquipmentDescription',
    'segmentsDurationInSeconds', 'segmentsDistance', 'segmentsCabinCode',
    'segmentsArrivalAirportCode', 'segmentsDepartureAirportCode',
    'segmentsDepartureTimeRaw'
]

# Inicializar codificadores globales por columna
encoders = {
    'airline': LabelEncoder(),
    'aircraft': LabelEncoder(),
    'cabin': LabelEncoder(),
    'arr_airport': LabelEncoder(),
    'dep_airport': LabelEncoder(),
}

# Diccionario para ir acumulando valores únicos por codificador
fitting_data = {key: set() for key in encoders}

def collect_categories(chunk):
    for col, key in zip([
        'segmentsAirlineCode', 'segmentsEquipmentDescription',
        'segmentsCabinCode', 'segmentsArrivalAirportCode', 'segmentsDepartureAirportCode'
    ], fitting_data):
        chunk[col].dropna().str.split('||').apply(fitting_data[key].update)

def fit_encoders():
    os.makedirs('../models', exist_ok=True)
    for key in encoders:
        encoders[key].fit(sorted(fitting_data[key]))
        joblib.dump(encoders[key], f"../models/encoder_{key}.pkl")

def explode_segments_chunk(chunk, chunk_index):
    exploded_segments = []

    for idx, (_, row) in enumerate(chunk.iterrows()):
        if idx % 100_000 == 0:
            print(f"Procesando fila {idx} del chunk {chunk_index}...")

        leg_id = row['legId']
        segment_data = {
            'segmentsAirlineCode': str(row['segmentsAirlineCode']).split('||') if pd.notnull(row['segmentsAirlineCode']) else [],
            'segmentsEquipmentDescription': str(row['segmentsEquipmentDescription']).split('||') if pd.notnull(row['segmentsEquipmentDescription']) else [],
            'segmentsDurationInSeconds': str(row['segmentsDurationInSeconds']).split('||') if pd.notnull(row['segmentsDurationInSeconds']) else [],
            'segmentsDistance': str(row['segmentsDistance']).split('||') if pd.notnull(row['segmentsDistance']) else [],
            'segmentsCabinCode': str(row['segmentsCabinCode']).split('||') if pd.notnull(row['segmentsCabinCode']) else [],
            'segmentsArrivalAirportCode': str(row['segmentsArrivalAirportCode']).split('||') if pd.notnull(row['segmentsArrivalAirportCode']) else [],
            'segmentsDepartureAirportCode': str(row['segmentsDepartureAirportCode']).split('||') if pd.notnull(row['segmentsDepartureAirportCode']) else [],
            'segmentsDepartureTimeRaw': str(row['segmentsDepartureTimeRaw']).split('||') if pd.notnull(row['segmentsDepartureTimeRaw']) else [],
        }

        max_len = max(len(segment_data[k]) for k in segment_data)

        for i in range(max_len):
            duration_val = segment_data['segmentsDurationInSeconds'][i] if i < len(segment_data['segmentsDurationInSeconds']) else None
            distance_val = segment_data['segmentsDistance'][i] if i < len(segment_data['segmentsDistance']) else None

            try:
                duration = float(duration_val) if duration_val not in [None, 'None', 'nan', ''] else None
            except:
                duration = None

            try:
                distance = float(distance_val) if distance_val not in [None, 'None', 'nan', ''] else None
            except:
                distance = None

            dep_time = segment_data['segmentsDepartureTimeRaw'][i] if i < len(segment_data['segmentsDepartureTimeRaw']) else None
            try:
                dep_hour = pd.to_datetime(dep_time).hour if dep_time else None
                dep_weekday = pd.to_datetime(dep_time).weekday() if dep_time else None
            except:
                dep_hour, dep_weekday = None, None

            exploded_segments.append({
                'legId': leg_id,
                'segment_num': i,
                'airline_code': encoders['airline'].transform([segment_data['segmentsAirlineCode'][i]])[0] if i < len(segment_data['segmentsAirlineCode']) else None,
                'aircraft_type': encoders['aircraft'].transform([segment_data['segmentsEquipmentDescription'][i]])[0] if i < len(segment_data['segmentsEquipmentDescription']) else None,
                'duration_seconds': duration,
                'distance_miles': distance,
                'departure_hour': dep_hour,
                'departure_weekday': dep_weekday,
                'cabin': encoders['cabin'].transform([segment_data['segmentsCabinCode'][i]])[0] if i < len(segment_data['segmentsCabinCode']) else None,
                'arr_airport': encoders['arr_airport'].transform([segment_data['segmentsArrivalAirportCode'][i]])[0] if i < len(segment_data['segmentsArrivalAirportCode']) else None,
                'dep_airport': encoders['dep_airport'].transform([segment_data['segmentsDepartureAirportCode'][i]])[0] if i < len(segment_data['segmentsDepartureAirportCode']) else None,
            })

    return pd.DataFrame(exploded_segments)

def run_labeled_segment_extraction(dataset_path):
    os.makedirs('../cleaned_label_segments', exist_ok=True)

    print("🔍 Primera pasada: recopilando categorías únicas...")
    for chunk in pd.read_csv(dataset_path, usecols=usecols, chunksize=1_000_000):
        collect_categories(chunk)
    fit_encoders()

    print("⚙️ Segunda pasada: explotando y codificando segmentos...")
    for i, chunk in enumerate(pd.read_csv(dataset_path, usecols=usecols, chunksize=1_000_000)):
        print(f"✈️ Etiquetando segmentos del chunk {i}...")
        exploded = explode_segments_chunk(chunk, i)
        exploded.to_parquet(f'../cleaned_label_segments/segment_chunk_{i}.parquet', index=False)
        print(f"✅ Segment chunk {i} guardado con {len(exploded)} segmentos")
