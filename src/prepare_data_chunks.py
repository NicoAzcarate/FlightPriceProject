import pandas as pd
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Ruta a los archivos .parquet ya divididos
chunks_path = "../data_subsets/final_dataset_chunks/final_chunk_*.parquet"
output_dir = "../data_subsets/prepared_chunks"
os.makedirs(output_dir, exist_ok=True)

print("📦 Cargando todos los chunks para preparar los datos de entrenamiento...")
all_chunks = [pd.read_parquet(file) for file in sorted(glob.glob(chunks_path))]
df = pd.concat(all_chunks, ignore_index=True)

print(f"✅ Dataset combinado: {df.shape}")

# Separar variables independientes (X) y la variable objetivo (y)
X = df.drop(columns=['totalFare'])
y = df['totalFare']

# División en entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Guardar los datos escalados y el escalador
pd.DataFrame(X_train_scaled, columns=X.columns).to_parquet(os.path.join(output_dir, "X_train.parquet"), index=False)
pd.DataFrame(X_test_scaled, columns=X.columns).to_parquet(os.path.join(output_dir, "X_test.parquet"), index=False)
pd.DataFrame(y_train).to_parquet(os.path.join(output_dir, "y_train.parquet"), index=False)
pd.DataFrame(y_test).to_parquet(os.path.join(output_dir, "y_test.parquet"), index=False)

joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

print("✅ Datos escalados y guardados para el entrenamiento.")
