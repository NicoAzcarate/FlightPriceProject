import pandas as pd
import os

def create_random_subset_chunked(input_path, output_path, frac=0.1, seed=42, chunksize=1_000_000):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    first = True
    total_rows = 0
    sampled_rows = 0
    reader = pd.read_csv(input_path, chunksize=chunksize)
    
    for i, chunk in enumerate(reader):
        total_rows += len(chunk)
        sample = chunk.sample(frac=frac, random_state=seed + i)
        sampled_rows += len(sample)
        
        sample.to_csv(output_path, index=False, mode='w' if first else 'a', header=first)
        first = False

        print(f"📦 Procesado chunk {i} - {len(chunk)} filas -> {len(sample)} seleccionadas")

    print(f"\n✅ Subset creado con {sampled_rows} de {total_rows} filas totales.")
