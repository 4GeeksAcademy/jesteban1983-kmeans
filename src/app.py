"""Entrenamiento reproducible del proyecto K-Means de California Housing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["Latitude", "Longitude", "MedInc"]
N_CLUSTERS = 6
RANDOM_STATE = 42
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "housing.csv"
MODELS_DIR = ROOT_DIR / "models"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def load_dataset(path: Path) -> pd.DataFrame:
	"""Carga el dataset y valida las variables exigidas por el ejercicio."""
	if not path.exists():
		raise FileNotFoundError(f"No se encontró el dataset: {path}")
	data = pd.read_csv(path)
	missing = set(FEATURES) - set(data.columns)
	if missing:
		raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
	if data[FEATURES].isna().any().any():
		raise ValueError("Las variables de clustering contienen valores nulos")
	return data


def build_models() -> tuple[Pipeline, RandomForestClassifier]:
	"""Construye K-Means escalado y el clasificador supervisado."""
	kmeans = Pipeline([
		("scaler", StandardScaler()),
		("kmeans", KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=20)),
	])
	classifier = RandomForestClassifier(
		n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
	)
	return kmeans, classifier


def save_cluster_plot(train: pd.DataFrame, test: pd.DataFrame, path: Path) -> None:
	"""Guarda la visualización geográfica de clusters y puntos de prueba."""
	fig, ax = plt.subplots(figsize=(11, 7))
	ax.scatter(train["Longitude"], train["Latitude"], c=train["cluster"], cmap="tab10", s=8, alpha=0.35, label="Entrenamiento")
	ax.scatter(test["Longitude"], test["Latitude"], c=test["cluster"], cmap="tab10", s=20, marker="x", alpha=0.8, label="Prueba")
	ax.set(title="Clusters de California Housing (K-Means, k=6)", xlabel="Longitud", ylabel="Latitud")
	ax.legend()
	fig.tight_layout()
	fig.savefig(path, dpi=150)
	plt.close(fig)


def run(data_path: Path = DEFAULT_DATA_PATH) -> dict:
	"""Ejecuta el pipeline, guarda artefactos y devuelve métricas."""
	data = load_dataset(data_path)
	train_x, test_x = train_test_split(data[FEATURES], test_size=0.2, random_state=RANDOM_STATE)
	kmeans, classifier = build_models()
	kmeans.fit(train_x)
	train_y, test_y = kmeans.predict(train_x), kmeans.predict(test_x)
	train_labeled, test_labeled = train_x.copy(), test_x.copy()
	train_labeled["cluster"], test_labeled["cluster"] = train_y, test_y
	classifier.fit(train_x, train_y)
	classifier_test_y = classifier.predict(test_x)
	metrics = {
		"n_rows": int(len(data)), "n_train": int(len(train_x)), "n_test": int(len(test_x)),
		"features": FEATURES, "n_clusters": N_CLUSTERS, "random_state": RANDOM_STATE,
		"kmeans_silhouette_train": float(silhouette_score(train_x, train_y)),
		"classifier_accuracy_vs_kmeans_test": float(accuracy_score(test_y, classifier_test_y)),
		"classifier_confusion_matrix": confusion_matrix(test_y, classifier_test_y).tolist(),
		"classification_report": classification_report(test_y, classifier_test_y, output_dict=True, zero_division=0),
		"train_cluster_counts": train_labeled["cluster"].value_counts().sort_index().to_dict(),
		"test_cluster_counts": test_labeled["cluster"].value_counts().sort_index().to_dict(),
	}
	MODELS_DIR.mkdir(exist_ok=True)
	PROCESSED_DIR.mkdir(exist_ok=True)
	joblib.dump(kmeans, MODELS_DIR / "kmeans_pipeline.joblib")
	joblib.dump(classifier, MODELS_DIR / "cluster_classifier.joblib")
	train_labeled.to_csv(PROCESSED_DIR / "housing_train_clustered.csv", index=False)
	test_labeled.to_csv(PROCESSED_DIR / "housing_test_clustered.csv", index=False)
	(PROCESSED_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
	save_cluster_plot(train_labeled, test_labeled, PROCESSED_DIR / "clusters.png")
	return metrics


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
	args = parser.parse_args()
	print(json.dumps(run(args.data), indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()
