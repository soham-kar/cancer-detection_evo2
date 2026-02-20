"""
Inspect Supplementary Table 2 from CIRCLE-seq paper.
"""
import pandas as pd
from pathlib import Path

FILE_PATH = Path('data/benchmark/circle_seq/Supplementary_Table_2.xlsx')

def inspect_excel():
    print(f"Loading {FILE_PATH}...")
    # Load all sheets to see what's inside
    xl = pd.ExcelFile(FILE_PATH)
    print(f"Sheets: {xl.sheet_names}")
    
    with open('cols.txt', 'w') as f:
        for sheet in xl.sheet_names:
            f.write(f"\n--- Sheet: {sheet} ---\n")
            df = pd.read_excel(FILE_PATH, sheet_name=sheet, nrows=5)
            f.write(f"Columns: {df.columns.tolist()}\n")
            for col in df.columns:
                f.write(f"  {repr(col)}\n")
    print("Wrote columns to cols.txt")

if __name__ == "__main__":
    inspect_excel()
