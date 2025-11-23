from pathlib import Path
import json

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import matplotlib

# backend headless para evitar problemas en CLI
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_STATE = 42


# ------------------------
# Utilidades de carga
# ------------------------

def _resolve_split_paths() -> tuple[Path, Path]:
    """
    Usa los splits limpios (fit/apply) generados sin leakage.
    """
    base_dir = Path(__file__).resolve().parents[1]
    train_path = base_dir / "SPLITS_FIT_APPLY" / "clean" / "train_clean.csv"
    test_path = base_dir / "SPLITS_FIT_APPLY" / "clean" / "test_clean.csv"
    if train_path.exists() and test_path.exists():
        return train_path, test_path
    raise FileNotFoundError("No se encontraron splits limpios en SPLITS_FIT_APPLY/clean.")


def load_data():
    """
    Carga train_clean y test_clean desde SPLITS_FIT_APPLY/clean.
    Deja is_attack como target y descarta Attack_type de las features.
    """
    train_path, test_path = _resolve_split_paths()

    print(f"[data] Leyendo train: {train_path}")
    print(f"[data] Leyendo test:  {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    cols_to_drop = ["is_attack", "Attack_type"]
    X_train = train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns])
    y_train = train_df["is_attack"].astype(int)

    X_test = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns])
    y_test = test_df["is_attack"].astype(int)

    print(f"[data] Train shape: {X_train.shape}")
    print(f"[data] Test  shape: {X_test.shape}")

    return X_train, y_train, X_test, y_test


# ------------------------
# Entrenamiento modelos
# ------------------------

def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """
    Entrena un Random Forest con hiperparámetros tunados previamente.
    """
    clf = RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=1,
        min_samples_split=2,
        max_features="sqrt",
    )
    print("[RF] Entrenando Random Forest con hiperparámetros fijos...")
    clf.fit(X_train, y_train)
    return clf


def train_bagging(X_train, y_train) -> BaggingClassifier:
    """
    Entrena un BaggingClassifier con un Árbol de Decisión base (hiperparámetros fijos).
    """
    base_tree = DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )

    clf = BaggingClassifier(
        estimator=base_tree,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        n_estimators=100,
        max_samples=1.0,
        max_features=1.0,
    )
    print("[BAG] Entrenando Bagging (DecisionTree) con hiperparámetros fijos...")
    clf.fit(X_train, y_train)
    return clf


# ------------------------
# Evaluación y guardado
# ------------------------

def save_confusion_matrix_plot(cm, classes, title: str, out_path: Path):
    """
    Guarda una imagen PNG con la matriz de confusion (en ingles).
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")  # paleta clara para mejor contraste
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

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


def evaluate_and_save(
    name: str,
    clf,
    X_test,
    y_test,
    metrics_subdir: str,
):
    """
    Evalua un modelo en test y guarda:
      - metricas en JSON
      - matriz de confusion en PNG
    """
    print(f"\n[eval-{name}] Evaluando modelo en test set...")

    y_pred = clf.predict(X_test)

    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_test)[:, 1]
        try:
            roc = roc_auc_score(y_test, y_proba)
        except Exception:
            roc = None
    else:
        y_proba = None
        roc = None

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    clf_report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["benign (0)", "attack (1)"],
        digits=4,
    )

    print(f"\n[eval-{name}] Classification report (test):")
    print(clf_report)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print(f"[eval-{name}] Confusion matrix (test):")
    print(cm)

    if roc is not None:
        print(f"[eval-{name}] ROC-AUC (test): {roc:.4f}")
    else:
        print(f"[eval-{name}] ROC-AUC not available.")

    base_dir = Path(__file__).resolve().parent
    metrics_dir = base_dir / metrics_subdir
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc) if roc is not None else None,
        "confusion_matrix": cm.tolist(),
        "classification_report_text": clf_report,
    }

    if name == "random_forest" and hasattr(clf, "feature_importances_"):
        feature_importances = dict(zip(X_test.columns, clf.feature_importances_.tolist()))
        metrics["feature_importances"] = feature_importances

    json_path = metrics_dir / f"{name}_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n[eval-{name}] Metricas guardadas en: {json_path}")

    cm_png_path = metrics_dir / f"{name}_confusion_matrix.png"
    save_confusion_matrix_plot(
        cm,
        classes=["0: benign", "1: attack"],
        title=f"Confusion Matrix - {name.replace('_', ' ').title()}",
        out_path=cm_png_path,
    )
    print(f"[eval-{name}] Confusion matrix saved at: {cm_png_path}")


# ------------------------
# Main
# ------------------------

def main():
    X_train, y_train, X_test, y_test = load_data()

    # Random Forest
    rf_clf = train_random_forest(X_train, y_train)
    evaluate_and_save("random_forest", rf_clf, X_test, y_test, "metrics_rf")

    # Bagging + Árbol
    bag_clf = train_bagging(X_train, y_train)
    evaluate_and_save("bagging_tree", bag_clf, X_test, y_test, "metrics_bagging")


if __name__ == "__main__":
    main()
