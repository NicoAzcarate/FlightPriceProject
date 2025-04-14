Mi proyecto tiene como objetivo desarrollar un sistema capaz de predecir el precio de un vuelo a partir de distintas características como origen, destino, aerolínea, clase, número de escalas, etc. A lo largo del desarrollo, he trabajado con un dataset de más de 80 millones de registros proporcionado por Expedia (a través de Kaggle), con el que he procesado mediante un pipeline completo de preprocesamiento, modelado y despliegue.



* He organizado el proyecto en las siguientes carpetas y archivos:

src/: scripts Python para limpiar, dividir, procesar y codificar los datos.
notebooks/: notebooks donde realizo exploración, preparación, entrenamiento y predicción.
models/: contiene los modelos entrenados (.h5) y los codificadores (.pkl).
data_subsets/: versiones intermedias y finales de los datos procesados (parquet, csv, npy...), con final_dataset.parquet como dataset final para el entrenamiento.
cleaned_chunks/ y cleaned_label_segments/: versiones codificadas por separado de los datos.



* Preparación y limpieza de los datos:

Primero trabajé con un subset del 10% del dataset original para poder manejarlo mejor y entender su estructura. Utilicé el script subset_creator.py para seleccionar y guardar este subset como itineraries_subset.csv.

Muchos itinerarios contenían múltiples segmentos (vuelos), así que desarrollé label_segments.py para separar estos segmentos y extraer características por tramo.

Usé el script clean_subset_chunks.py para dividir el subset en chunks más pequeños, codificar algunas columnas categóricas y guardar los datos como parquet. Además genera un archivo column_values.json con los mapeos originales de categorías.

Una vez codificados por separado los itinerarios (cleaned_chunks) y los segmentos (cleaned_label_segments), los combiné utilizando el campo legId, generando el archivo final_dataset.parquet. A partir de este dataset, creé archivos divididos (final_dataset_chunks) y conjuntos escalados para entrenamiento y test en train_test usando prepare_data_chunks.py.



* Modelo:

En ModelTraining.ipynb probé diferentes arquitecturas de redes neuronales con Keras: modelos con más o menos capas ocultas, funciones de activación ReLU y lineal, variaciones en el learning rate y regularización...

Tras varias pruebas, el mejor rendimiento lo obtuve con el modelo flight_price_predictor_v4_clean_final.h5, entrenado sobre el dataset limpio y combinado. Y lo guardé en la carpeta models.

En el notebook exploracion_columnas.ipynb realicé un análisis de las columnas, identificando qué variables utilizar como input en el modelo, su distribución y correlación con el precio (totalFare).

En Predictions.ipynb validé el modelo con nuevos datos y visualicé los resultados. Además, desarrollé app_predictor.py, un script que uso como backend para una interfaz creada con Streamlit, donde el usuario puede introducir las características del vuelo y obtener la predicción del precio.



Tecnología usada:

- Python + Pandas, NumPy, Scikit-learn
- TensorFlow / Keras
- Streamlit para la interfaz web
- Jupyter Notebooks para desarrollo exploratorio
- VS Code como entorno de desarrollo
- Ubuntu como sistema operativo principal