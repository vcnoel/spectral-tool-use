# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

#!/usr/bin/env python3
"""
Compare feature distributions between calibration and validation sets.
"""

import pandas as pd
import numpy as np

# Load both datasets
calib_df = pd.read_csv("data/qwen_sweep_calib_50.csv")
val_df = pd.read_csv("data/qwen_sweep_validation_2300.csv")

print("=== Distribution Comparison ===\n")

# Check a few key features
features = ["L7_Fiedler", "L21_Smoothness", "L1_HFER"]

for feat in features:
    if feat in calib_df.columns and feat in val_df.columns:
        print(f"{feat}:")
        print(f"  Calib - Mean: {calib_df[feat].mean():.6f}, Std: {calib_df[feat].std():.6f}")
        print(f"  Valid - Mean: {val_df[feat].mean():.6f}, Std: {val_df[feat].std():.6f}")
        
        # Check correlation with label
        calib_corr = calib_df[[feat, 'is_hallucination']].corr().iloc[0, 1]
        val_corr = val_df[[feat, 'is_hallucination']].corr().iloc[0, 1]
        print(f"  Calib Correlation: {calib_corr:.4f}")
        print(f"  Valid Correlation: {val_corr:.4f}")
        print()

# Class balance
print("\n=== Class Balance ===")
print(f"Calibration: {calib_df['is_hallucination'].sum()}/{len(calib_df)} = {calib_df['is_hallucination'].mean():.2%}")
print(f"Validation:  {val_df['is_hallucination'].sum()}/{len(val_df)} = {val_df['is_hallucination'].mean():.2%}")
