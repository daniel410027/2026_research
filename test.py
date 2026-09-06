import pandas as pd

input_path = "20260331_shap_summary.csv"
output_path = "features_only.csv"

df = pd.read_csv(input_path, encoding="utf-8-sig")  # utf-8-sig 處理 BOM (﻿)
df[["feature"]].to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"Done: {len(df)} features saved to {output_path}")