# California Housing: K-Means y clasificación supervisada

Este proyecto aplica aprendizaje no supervisado y supervisado al conjunto de datos California Housing. Agrupa viviendas utilizando latitud, longitud e ingreso medio (`MedInc`) y después entrena un clasificador para reproducir las etiquetas generadas por K-Means.

## Flujo implementado

1. Carga y validación de `data/housing.csv`.
2. División reproducible en entrenamiento (80 %) y prueba (20 %).
3. Estandarización de `Latitude`, `Longitude` y `MedInc`.
4. Entrenamiento de K-Means con 6 clusters, `random_state=42` y `n_init=20`.
5. Predicción de clusters para train y test.
6. Entrenamiento de `RandomForestClassifier` con las etiquetas de K-Means de train.
7. Guardado de métricas, datasets etiquetados, gráfico y modelos serializados.

## Estructura

```text
├── data/housing.csv
├── data/processed/ (CSV etiquetados, metrics.json y clusters.png)
├── models/ (kmeans_pipeline.joblib y cluster_classifier.joblib)
├── src/app.py
└── requirements.txt
```

## Instalación y ejecución

Se recomienda Python 3.11 o superior:

```bash
pip install -r requirements.txt
python src/app.py
```

También puede indicarse otro CSV con `python src/app.py --data ruta/al/housing.csv`. El comando muestra las métricas y actualiza los artefactos en `models/` y `data/processed/`.

## Resultados

Con `random_state=42`, el pipeline procesa 20.640 registros (16.512 de train y 4.128 de test), obtiene un silhouette score de train aproximado de **0,394** y alcanza una exactitud aproximada de **0,995** al reproducir el clasificador supervisado las etiquetas de K-Means en test. Esta exactitud mide el acuerdo con etiquetas generadas automáticamente, no con una verdad externa etiquetada.

`data/processed/clusters.png` representa longitud y latitud, colorea los puntos según el cluster de K-Means y marca los puntos de test con una `x`. Los clusters dependen principalmente de la posición geográfica, con una contribución adicional de `MedInc`.

## Modelos guardados

- `models/kmeans_pipeline.joblib`: pipeline de `StandardScaler` + `KMeans`.
- `models/cluster_classifier.joblib`: Random Forest supervisado.

```python
import joblib

kmeans = joblib.load("models/kmeans_pipeline.joblib")
classifier = joblib.load("models/cluster_classifier.joblib")
new_points = [[34.05, -118.25, 4.5]]  # Latitude, Longitude, MedInc
print(kmeans.predict(new_points))
print(classifier.predict(new_points))
```
