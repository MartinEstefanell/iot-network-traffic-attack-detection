
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
csv_to_xlsx_format.py
---------------------
Convierte un CSV a XLSX con formato 'tipo clean_DataSet':
- Fila de encabezados en negrita.
- Filtro automático.
- Encabezado congelado (freeze panes).
- Ancho de columnas auto-ajustado según contenido.
- Estilo de tabla con filas bandeadas (verde claro).
- No altera los datos (no limpia ni normaliza).

Uso:
  python csv_to_xlsx_format.py input.csv [output.xlsx] [--sheet-name Hoja1] [--table-style Light9]

Notas:
- Requiere: pandas, xlsxwriter
"""

import sys
import os
import argparse
import pandas as pd
from pathlib import Path

DEFAULT_SHEET = "Sheet1"
DEFAULT_STYLE = "Table Style Light 9"  # verde claro

def autosize_columns(ws, df):
    for i, col in enumerate(df.columns):
        max_len = max([len(str(col))] + [len(str(v)) for v in df[col].astype(str).values])
        ws.set_column(i, i, min(max_len + 2, 60))

def convert_csv_to_formatted_xlsx(input_csv: str, output_xlsx: str, sheet_name: str = DEFAULT_SHEET, table_style: str = DEFAULT_STYLE):
    # Leer CSV tal cual
    df = pd.read_csv(input_csv, low_memory=False)

    # Escribir con formato
    with pd.ExcelWriter(output_xlsx, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        wb = writer.book
        ws = writer.sheets[sheet_name]

        # Formato encabezado
        header_fmt = wb.add_format({"bold": True, "valign": "top"})
        for col_idx, name in enumerate(df.columns):
            ws.write(0, col_idx, name, header_fmt)

        # Freeze header
        ws.freeze_panes(1, 0)

        # Crear tabla para banded rows (verde claro)
        nrows, ncols = df.shape
        if ncols > 0 and nrows >= 0:
            columns = [{"header": str(col)} for col in df.columns]
            ws.add_table(0, 0, nrows, ncols - 1, {
                "columns": columns,
                "style": table_style
            })

        # Auto ancho de columnas
        autosize_columns(ws, df)

    return output_xlsx


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Convertir CSV a XLSX con formato tipo clean_DataSet.")
    p.add_argument("input_csv", help="Ruta al archivo .csv de entrada")
    p.add_argument("output_xlsx", nargs="?", help="Ruta al .xlsx de salida (opcional). Si no se indica, se usa el mismo nombre con extensión .xlsx")
    p.add_argument("--outdir", help="Directorio donde guardar el .xlsx de salida (opcional). Si se indica junto con output_xlsx, se ignora.)")
    p.add_argument("--sheet-name", default=DEFAULT_SHEET, help="Nombre de la hoja (por defecto: Sheet1)")
    p.add_argument("--table-style", default=DEFAULT_STYLE, help="Estilo de tabla de Excel (por defecto: 'Table Style Light 9')")
    return p.parse_args(argv)

def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise SystemExit(f"No se encontró el CSV: {input_csv}")
    # Determinar ruta de salida: prioridad -> output_xlsx explicito, luego --outdir, luego mismo directorio del CSV
    if args.output_xlsx:
        output_xlsx = args.output_xlsx
    elif args.outdir:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        output_xlsx = str(outdir / (input_csv.stem + ".xlsx"))
    else:
        output_xlsx = str(input_csv.with_suffix(".xlsx"))
    convert_csv_to_formatted_xlsx(str(input_csv), output_xlsx, sheet_name=args.sheet_name, table_style=args.table_style)
    print(f"[OK] XLSX generado: {output_xlsx}")


if __name__ == "__main__":
    main()
