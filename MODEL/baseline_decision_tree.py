from pathlib import Path
import json

import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import matplotlib.pyplot as plt

RANDOM_STATE = 42


def _resolve_split_paths() -> tuple[Path, Path]:
    """
    Prioriza los splits limpios (fit/apply). Si no existen, cae al SPLITS original.
    """
    base_dir = Path(__file__).resolve().parents[1]  # sube un nivel desde MODEL/
    candidates = [
        (
            base_dir / "SPLITS_FIT_APPLY" / "clean" / "train_clean.csv",
            base_dir / "SPLITS_FIT_APPLY" / "clean" / "test_clean.csv",
        ),
    ]

    for tr, te in candidates:
        if tr.exists() and te.exists():
            return tr, te

    raise FileNotFoundError(
        "No se encontraron splits. Genera nuevos con fit_apply_pipeline.py")


def load_data():
    """
    Carga train/test; prioriza los splits limpios en SPLITS_FIT_APPLY/clean.
    """
    train_path, test_path = _resolve_split_paths()

    print(f"[data] Leyendo train: {train_path}")
    print(f"[data] Leyendo test:  {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # sacamos columnas que NO deben ser features
    cols_to_drop = ["is_attack", "Attack_type"]
    X_train = train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns])
    y_train = train_df["is_attack"].astype(int)

    X_test = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns])
    y_test = test_df["is_attack"].astype(int)

    print(f"[data] Train shape: {X_train.shape}")
    print(f"[data] Test  shape: {X_test.shape}")

    return X_train, y_train, X_test, y_test


def train_baseline_decision_tree(X_train, y_train):
    """
    Entrena un Arbol de Decision como baseline usando GridSearchCV
    sobre el conjunto de entrenamiento (con cross-validation estratificada).
    """
    base_clf = DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",  # importante por el desbalance
    )

    param_grid = {
        "max_depth": [3, 5, 7, None],
        "min_samples_leaf": [1, 5, 10],
        "min_samples_split": [2, 10, 20],
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    grid = GridSearchCV(
        estimator=base_clf,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    print("[model] Iniciando GridSearchCV (DecisionTree baseline)...")
    grid.fit(X_train, y_train)

    print("\n[model] Mejores hiperparametros encontrados:")
    print(grid.best_params_)

    print(f"\n[model] Mejor score de CV (f1_macro): {grid.best_score_:.4f}")

    best_clf = grid.best_estimator_
    return best_clf


def save_confusion_matrix_plot(cm, classes, out_path: Path):
    """
    Guarda una imagen PNG con la matriz de confusion (en ingles para reportes).
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")  # paleta clara para mejor contraste
    ax.set_title("Confusion Matrix - Baseline Decision Tree")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    # Rotar etiquetas del eje X
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Escribir los valores dentro de las celdas
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_model(clf, X_test, y_test):
    """
    Evalua el modelo final sobre el conjunto de test y:
      - imprime metricas
      - guarda metricas en JSON
      - guarda plot de matriz de confusion como PNG
    """
    print("\n[eval] Evaluando modelo en test set...")

    y_pred = clf.predict(X_test)

    # Metricas globales (sin desglosar por clase)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    clf_report = classification_report(
        y_test, y_pred, labels=[0, 1], target_names=["benigno (0)", "ataque (1)"], digits=4
    )

    print("\n[eval] Classification report (test):")
    print(clf_report)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print("[eval] Matriz de confusion (test):")
    print(cm)

    # ---- Guardar metricas en JSON + matriz en PNG ----
    base_dir = Path(__file__).resolve().parent
    metrics_dir = base_dir / "metrics_baseline"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "classification_report_text": clf_report,
    }

    json_path = metrics_dir / "dt_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n[eval] Metricas guardadas en: {json_path}")

    # Plot de la matriz de confusion
    cm_png_path = metrics_dir / "dt_confusion_matrix.png"
    save_confusion_matrix_plot(cm, classes=["0: benign", "1: attack"], out_path=cm_png_path)
    print(f"[eval] Matriz de confusion guardada en: {cm_png_path}")


def main():
    X_train, y_train, X_test, y_test = load_data()
    clf = train_baseline_decision_tree(X_train, y_train)
    evaluate_model(clf, X_test, y_test)


if __name__ == "__main__":
    main()
