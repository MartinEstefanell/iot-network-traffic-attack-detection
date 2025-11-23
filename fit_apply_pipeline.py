"""
Pipeline helper: split raw dataset first, then fit cleaning on train and apply same artifacts on test.
Outputs raw splits and cleaned splits without leaking test info into artifact fitting.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

# Reuse existing helpers
from split_data import guess_label_col, derive_is_attack
from CLEANING.cleaning import load_cleaning_modules, topo_sort, finalize_dataset

DEFAULT_ARTIFACTS_DIR = Path("CLEANING/.artifacts_fit_train")
DEFAULT_RAW_OUTDIR = Path("SPLITS_FIT_APPLY/raw")
DEFAULT_CLEAN_OUTDIR = Path("SPLITS_FIT_APPLY/clean")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Split raw dataset, fit cleaning on train, apply on test.")
    ap.add_argument("--input", help="Ruta al dataset crudo (CSV/XLSX).", default=None)
    ap.add_argument("--train", type=float, default=0.80, help="Proporcion de train (default 0.80)")
    ap.add_argument("--test", type=float, default=0.20, help="Proporcion de test (default 0.20)")
    ap.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad")
    ap.add_argument("--raw-outdir", default=str(DEFAULT_RAW_OUTDIR), help="Carpeta para splits crudos")
    ap.add_argument("--clean-outdir", default=str(DEFAULT_CLEAN_OUTDIR), help="Carpeta para splits limpios")
    ap.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR), help="Carpeta donde guardar artefactos nuevos")
    return ap.parse_args()


def default_input_path() -> Path:
    """Busca dataset crudo razonable si no se pasa --input."""
    candidates = [
        Path("CLEANING/dataset_RT-IoT2022.csv"),
        Path("CLEANING/dataset_RT-IoT2022.xlsx"),
        Path("dataset_RT-IoT2022.csv"),
        Path("dataset_RT-IoT2022.xlsx"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("No se encontro dataset de entrada; pase --input explicitamente.")


def read_raw_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def split_raw(df: pd.DataFrame, train_p: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    label_col = guess_label_col(df)
    df_lab = derive_is_attack(df.copy(), col=label_col)
    df_unknown = df_lab[df_lab["is_attack"].isna()].copy()
    df_labeled = df_lab[~df_lab["is_attack"].isna()].copy()
    df_labeled["is_attack"] = df_labeled["is_attack"].astype(int)

    y = df_labeled["is_attack"]
    tr_df, te_df = train_test_split(
        df_labeled, test_size=1 - train_p, random_state=seed, stratify=y
    )

    report = {
        "rows_total": int(len(df)),
        "rows_labeled": int(len(df_labeled)),
        "rows_unknown": int(len(df_unknown)),
        "train": {"rows": int(len(tr_df)), "is_attack_counts": y.loc[tr_df.index].value_counts().to_dict()},
        "test": {"rows": int(len(te_df)), "is_attack_counts": y.loc[te_df.index].value_counts().to_dict()},
    }
    return tr_df, te_df, report


def _set_artifact_paths(stages: Dict[str, Any], artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for inst in stages.values():
        # Update instance and class attributes if present
        if hasattr(inst, "ARTIFACTS_DIR"):
            inst.ARTIFACTS_DIR = str(artifacts_dir)
            if hasattr(inst.__class__, "ARTIFACTS_DIR"):
                inst.__class__.ARTIFACTS_DIR = str(artifacts_dir)
        if hasattr(inst, "ARTIFACTS_FILE"):
            fname = Path(getattr(inst, "ARTIFACTS_FILE")).name
            new_path = artifacts_dir / fname
            inst.ARTIFACTS_FILE = str(new_path)
            if hasattr(inst.__class__, "ARTIFACTS_FILE"):
                inst.__class__.ARTIFACTS_FILE = str(new_path)


def run_cleaning(df: pd.DataFrame, artifacts_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    stages = load_cleaning_modules()
    _set_artifact_paths(stages, artifacts_dir)
    order = topo_sort(stages)
    cur = df.copy()
    reports: Dict[str, Any] = {"order": order, "modules": {}}

    for name in order:
        inst = stages[name]
        cur, rep = inst.run(cur)
        reports["modules"][name] = rep

    cur, final_rep = finalize_dataset(cur)
    reports["finalize"] = final_rep
    return cur, reports


def add_label_column(df: pd.DataFrame, y_true: pd.Series | None = None) -> pd.DataFrame:
    """
    Anade columna is_attack. Si se pasa y_true, se usa directamente.
    """
    df = df.copy()
    if y_true is None:
        label_col = guess_label_col(df)
        df = derive_is_attack(df, col=label_col)
        df = df[~df["is_attack"].isna()].copy()
        df["is_attack"] = df["is_attack"].astype(int)
        return df

    # usar labels externos (p.ej. aplicar sin revelar etiqueta durante cleaning)
    df["is_attack"] = y_true.astype(int)
    return df


def save_reports(reports: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)


def _mask_labels_for_apply(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evita que la limpieza en apply use la etiqueta: fuerza attack a unknown y borra is_attack.
    """
    df = df.copy()
    for col in ("attack_type", "Attack_type"):
        if col in df.columns:
            df[col] = "unknown"
    if "is_attack" in df.columns:
        df["is_attack"] = pd.NA
    return df


def main():
    args = parse_args()
    if round(args.train + args.test, 8) != 1.0:
        raise ValueError("Las proporciones deben sumar 1.0 (train + test)")

    input_path = Path(args.input) if args.input else default_input_path()
    raw_outdir = Path(args.raw_outdir)
    clean_outdir = Path(args.clean_outdir)
    artifacts_dir = Path(args.artifacts_dir)

    print(f"[data] Leyendo dataset crudo: {input_path}")
    raw_df = read_raw_dataset(input_path)
    print(f"[data] Filas totales: {len(raw_df)}")

    train_raw, test_raw, split_rep = split_raw(raw_df, train_p=args.train, seed=args.seed)
    raw_outdir.mkdir(parents=True, exist_ok=True)
    train_raw.to_csv(raw_outdir / "train_raw.csv", index=False)
    test_raw.to_csv(raw_outdir / "test_raw.csv", index=False)
    if split_rep.get("rows_unknown", 0) > 0:
        # Guardamos unknown aparte por transparencia
        unknown = raw_df.loc[raw_df.index.difference(train_raw.index).difference(test_raw.index)]
        unknown.to_csv(raw_outdir / "unknown_raw.csv", index=False)
    save_reports({"split": split_rep}, raw_outdir / "split_report.json")
    print(f"[split] Guardado raw train/test en {raw_outdir}")

    # ---- Fit cleaning on train ----
    print("[clean] Ajustando limpieza sobre train...")
    clean_train, rep_train = run_cleaning(train_raw, artifacts_dir)
    clean_train = add_label_column(clean_train)

    print("[clean] Aplicando limpieza sobre test (usando mismos artefactos, sin exponer etiqueta)...")
    # guardamos label real y limpiamos con attack desconocido para evitar leakage
    label_col = guess_label_col(test_raw)
    y_true = derive_is_attack(test_raw.copy(), col=label_col)["is_attack"]
    test_for_clean = _mask_labels_for_apply(test_raw)
    clean_test, rep_test = run_cleaning(test_for_clean, artifacts_dir)
    clean_test = add_label_column(clean_test, y_true=y_true)

    clean_outdir.mkdir(parents=True, exist_ok=True)
    clean_train.to_csv(clean_outdir / "train_clean.csv", index=False)
    clean_test.to_csv(clean_outdir / "test_clean.csv", index=False)

    save_reports({"train": rep_train, "test": rep_test}, clean_outdir / "cleaning_reports.json")
    print(f"[clean] Guardado train/test limpios en {clean_outdir}")
    print(f"[clean] Artefactos en {artifacts_dir}")


if __name__ == "__main__":
    main()
