# -*- coding: utf-8 -*-
"""
EDA - Intensidad y Tamaño de Flujo (RT-IoT2022)
-----------------------------------------------
- Compatible con EDA.py (main(df, out_dir)).
- Usa columnas EXACTAS del dataset:
  flow_duration, flow_pkts_payload.tot, fwd_pkts_tot, bwd_pkts_tot,
  payload_bytes_per_second, Attack_type.
- Exporta tablas a XLSX usando csv_to_xlsx_format.py.
- Genera figuras PNG (matplotlib, 1 por gráfico, sin seaborn).

Cubre:
1) Distribuciones (hist log de tot_bytes, box tot_pkts por clase)
2) Relación tamaño–duración (scatter) + correlaciones
3) Diferencias por clase (mean, median, std) + box para flow_byts_s
4) Coeficiente de variación (CV) por clase
5) Outliers (IQR y z-score) con % por clase
"""

import os
import sys
import subprocess
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NORMAL_HINTS = {"normal", "benign", "mqtt_publish"}

# -------------------- Utilidades --------------------
def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def norm_cat(s: pd.Series) -> pd.Series:
    if s is None:
        return s
    return s.astype(str).str.strip().str.lower().replace({"": np.nan})

def to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def is_normal_label(x: str) -> bool:
    if not isinstance(x, str):
        return False
    x = x.strip().lower()
    return any(h in x for h in NORMAL_HINTS)

def _find_converter() -> str | None:
    here = os.path.dirname(__file__)
    candidates = [
        os.path.join(here, "csv_to_xlsx_format.py"),
        os.path.join(os.path.dirname(here), "csv_to_xlsx_format.py"),
        os.path.join(os.getcwd(), "csv_to_xlsx_format.py"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return "csv_to_xlsx_format"  # En PATH

def _save_via_converter(df: pd.DataFrame, out_dir: str, base_name: str, sheet_name: str = "Sheet1") -> str:
    ensure_outdir(out_dir)
    converter = _find_converter()
    if converter is None:
        raise RuntimeError("No se encontró csv_to_xlsx_format.py")
    with tempfile.TemporaryDirectory() as tmpd:
        tmp_csv = os.path.join(tmpd, f"{base_name}.csv")
        df.to_csv(tmp_csv, index=False)
        xlsx_out = os.path.join(out_dir, f"{base_name}.xlsx")
        cmd = [sys.executable, converter, tmp_csv, xlsx_out, "--sheet-name", sheet_name, "--outdir", out_dir]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"[XLSX] {xlsx_out}")
        return xlsx_out

def _save_plot(fig, out_dir: str, name: str):
    ensure_outdir(out_dir)
    out = os.path.join(out_dir, name)
    fig.tight_layout(pad=1.0)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"[PNG] {out}")
    plt.close(fig)

# -------------------- Análisis --------------------
def main(df: pd.DataFrame, out_dir: str | None = None):
    """
    Bloque EDA: Intensidad y Tamaño de Flujo.
    """
    out_dir = out_dir or "Outputs_intensity_size"
    ensure_outdir(out_dir)

    # ---- Columnas exactas del dataset ----
    duration_col = "flow_duration"
    tot_bytes_col = "flow_pkts_payload.tot"     # bytes útiles totales del flujo
    fwd_pkts_col = "fwd_pkts_tot"
    bwd_pkts_col = "bwd_pkts_tot"
    flow_byts_s_col = "payload_bytes_per_second"
    attack_col = "Attack_type"

    # Validar presencia mínima
    required = [duration_col, tot_bytes_col, fwd_pkts_col, bwd_pkts_col, flow_byts_s_col, attack_col]
    for col in required:
        if col not in df.columns:
            print(f"⚠️  Columna no encontrada: {col}")

    # --- Casting / columnas derivadas ---
    df = df.copy()
    df[duration_col] = to_numeric_safe(df[duration_col])
    df[tot_bytes_col] = to_numeric_safe(df[tot_bytes_col])
    df[fwd_pkts_col] = to_numeric_safe(df[fwd_pkts_col])
    df[bwd_pkts_col] = to_numeric_safe(df[bwd_pkts_col])
    df[flow_byts_s_col] = to_numeric_safe(df[flow_byts_s_col])

    # tot_pkts = fwd + bwd
    df["__tot_pkts"] = df[fwd_pkts_col].fillna(0) + df[bwd_pkts_col].fillna(0)
    tot_pkts_col = "__tot_pkts"

    # Etiqueta (normal/attack)
    df[attack_col] = norm_cat(df[attack_col])
    df["is_attack"] = ~df[attack_col].apply(is_normal_label)  # True=attack, False=normal, NaN=unknown

    # ---------------- 1) Distribución general ----------------
    # Histograma log10(tot_bytes)
    tb = df[tot_bytes_col].dropna()
    if not tb.empty:
        fig = plt.figure()
        plt.hist(np.log10(tb + 1.0), bins=60)
        plt.xlabel("log10(Total Bytes + 1)")
        plt.ylabel("Count")
        plt.title("Figure A. Total Bytes (log10)", fontsize=11)
        _save_plot(fig, out_dir, "eda_tot_bytes_hist_log.png")

    # Boxplot tot_pkts por clase
    bp = df[[tot_pkts_col, "is_attack"]].dropna()
    if not bp.empty and bp["is_attack"].notna().any():
        fig = plt.figure()
        plt.boxplot(
            [bp.loc[bp["is_attack"] == False, tot_pkts_col],
             bp.loc[bp["is_attack"] == True, tot_pkts_col]],
            labels=["Normal", "Attack"]
        )
        plt.ylabel("Total Packets")
        plt.title("Figure B. Total Packets by Class", fontsize=11)
        _save_plot(fig, out_dir, "eda_tot_pkts_box_by_class.png")

    # ---------------- 2) Tamaño vs duración + correlaciones ----------------
    # Scatter (duration vs tot_bytes), coloreado por clase (marcadores)
    scatter_df = df[[duration_col, tot_bytes_col, "is_attack"]].dropna()
    if not scatter_df.empty:
        fig = plt.figure()
        # Normal
        sub_n = scatter_df[scatter_df["is_attack"] == False]
        if not sub_n.empty:
            plt.scatter(sub_n[duration_col], sub_n[tot_bytes_col], s=8, alpha=0.5, marker="o", label="Normal")
        # Attack
        sub_a = scatter_df[scatter_df["is_attack"] == True]
        if not sub_a.empty:
            plt.scatter(sub_a[duration_col], sub_a[tot_bytes_col], s=8, alpha=0.5, marker="x", label="Attack")
        # Unknown
        sub_u = scatter_df[scatter_df["is_attack"].isna()]
        if not sub_u.empty:
            plt.scatter(sub_u[duration_col], sub_u[tot_bytes_col], s=8, alpha=0.3, marker=".", label="Unknown")
        if any([not sub_n.empty, not sub_a.empty, not sub_u.empty]):
            plt.legend()
        plt.xlabel("Flow Duration (s)")
        plt.ylabel("Total Bytes (payload)")
        plt.title("Figure C. Duration vs Total Bytes", fontsize=11)
        _save_plot(fig, out_dir, "eda_scatter_duration_vs_tot_bytes.png")

    # Correlaciones entre variables clave
    corr_vars = [duration_col, tot_bytes_col, tot_pkts_col, flow_byts_s_col]
    corr_mat = df[corr_vars].corr(numeric_only=True)
    _save_via_converter(corr_mat.reset_index(), out_dir, "eda_correlations", sheet_name="correlations")

    # ---------------- 3) Diferencias entre clases ----------------
    if df["is_attack"].notna().any():
        stats = (
            df.groupby("is_attack")[[duration_col, tot_bytes_col, tot_pkts_col, flow_byts_s_col]]
            .agg(["mean", "median", "std"])
            .rename(index={False: "normal", True: "attack"})
        )
        _save_via_converter(stats.reset_index(), out_dir, "eda_classwise_stats", sheet_name="class_stats")

        # Boxplot para flow_byts_s por clase
        bx = df[[flow_byts_s_col, "is_attack"]].dropna()
        if not bx.empty:
            fig = plt.figure()
            plt.boxplot(
                [bx.loc[bx["is_attack"] == False, flow_byts_s_col],
                 bx.loc[bx["is_attack"] == True, flow_byts_s_col]],
                labels=["Normal", "Attack"]
            )
            plt.ylabel("Payload Bytes/s")
            plt.title("Figure D. Flow Bytes/s by Class", fontsize=11)
            _save_plot(fig, out_dir, "eda_flow_byts_s_box_by_class.png")

    # ---------------- 4) Coeficiente de variación (CV) ----------------
    def cv(s: pd.Series) -> float:
        m = s.mean()
        return float("nan") if m == 0 else s.std(ddof=1) / m

    if df["is_attack"].notna().any():
        cv_tab = (
            df.groupby("is_attack")[[duration_col, tot_bytes_col, tot_pkts_col, flow_byts_s_col]]
            .agg(cv)
            .rename(index={False: "normal", True: "attack"})
            .reset_index()
        )
        _save_via_converter(cv_tab, out_dir, "eda_classwise_cv", sheet_name="class_cv")

    # ---------------- 5) Outliers (IQR y z-score) ----------------
    rep_rows = []
    for var in [duration_col, tot_bytes_col, tot_pkts_col, flow_byts_s_col]:
        s = df[var].dropna()
        if s.empty:
            continue
        q1, q3 = np.percentile(s, [25, 75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mu, sd = s.mean(), s.std(ddof=1)

        # % por clase (normal / attack / unknown)
        label = df["is_attack"].map({False: "normal", True: "attack"}).fillna("unknown")
        for cls in ["normal", "attack", "unknown"]:
            idx = (label == cls) & df[var].notna()
            n = int(idx.sum())
            if n == 0:
                rep_rows.append({
                    "variable": var, "class": cls, "n": 0,
                    "iqr_outliers_%": np.nan, "zscore_outliers_%": np.nan,
                    "low_IQR": low, "high_IQR": high, "mean": mu, "std": sd
                })
                continue
            vals = df.loc[idx, var].astype(float)
            iqr_mask = (vals < low) | (vals > high)
            z_mask = (np.abs((vals - mu) / (sd + 1e-9)) > 3.0)
            rep_rows.append({
                "variable": var,
                "class": cls,
                "n": n,
                "iqr_outliers_%": round(iqr_mask.mean() * 100, 3),
                "zscore_outliers_%": round(z_mask.mean() * 100, 3),
                "low_IQR": low, "high_IQR": high,
                "mean": mu, "std": sd
            })

    outliers = pd.DataFrame(rep_rows)
    if not outliers.empty:
        _save_via_converter(outliers, out_dir, "eda_outliers_report", sheet_name="outliers")

    print(f"\n✅ intensity_size listo. Salidas en: {os.path.abspath(out_dir)}")
