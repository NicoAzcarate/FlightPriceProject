import pandas as pd

usecols = [
    'legId', 'segmentsAirlineCode', 'segmentsEquipmentDescription',
    'segmentsDurationInSeconds', 'segmentsDistance',
    'segmentsCabinCode', 'segmentsArrivalAirportCode'
]

def explode_segments_chunk(chunk):
    exploded_segments = []

    for _, row in chunk.iterrows():
        leg_id = row['legId']
        segment_data = {
            'segmentsAirlineCode': str(row['segmentsAirlineCode']).split('||') if pd.notnull(row['segmentsAirlineCode']) else [],
            'segmentsEquipmentDescription': str(row['segmentsEquipmentDescription']).split('||') if pd.notnull(row['segmentsEquipmentDescription']) else [],
            'segmentsDurationInSeconds': str(row['segmentsDurationInSeconds']).split('||') if pd.notnull(row['segmentsDurationInSeconds']) else [],
            'segmentsDistance': str(row['segmentsDistance']).split('||') if pd.notnull(row['segmentsDistance']) else [],
            'segmentsCabinCode': str(row['segmentsCabinCode']).split('||') if pd.notnull(row['segmentsCabinCode']) else [],
            'segmentsArrivalAirportCode': str(row['segmentsArrivalAirportCode']).split('||') if pd.notnull(row['segmentsArrivalAirportCode']) else []
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

            exploded_segments.append({
                'legId': leg_id,
                'segment_num': i,
                'airline': segment_data['segmentsAirlineCode'][i] if i < len(segment_data['segmentsAirlineCode']) else None,
                'aircraft': segment_data['segmentsEquipmentDescription'][i] if i < len(segment_data['segmentsEquipmentDescription']) else None,
                'duration': duration,
                'distance': distance,
                'cabin': segment_data['segmentsCabinCode'][i] if i < len(segment_data['segmentsCabinCode']) else None,
                'arr_airport': segment_data['segmentsArrivalAirportCode'][i] if i < len(segment_data['segmentsArrivalAirportCode']) else None,
            })

    return pd.DataFrame(exploded_segments)


def run_segment_extraction(dataset_path):
    import os
    os.makedirs('cleaned_segments', exist_ok=True)

    for i, chunk in enumerate(pd.read_csv(dataset_path, usecols=usecols, chunksize=10_000_000)):
        print(f"✈️ Extrayendo segmentos del chunk {i}...")
        exploded = explode_segments_chunk(chunk)
        exploded.to_parquet(f'../cleaned_segments/segment_chunk_{i}.parquet', index=False)
        print(f"✅ Segment chunk {i} guardado con {len(exploded)} segmentos")
