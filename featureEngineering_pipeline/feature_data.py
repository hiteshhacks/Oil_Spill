import pandas as pd
import numpy as np

# Load cleaned dataset
df = pd.read_csv("matched_ais_clean.csv")

print("Records:", len(df))
print("Vessels:", df["mmsi"].nunique())


# ==========================================
# FEATURE ENGINEERING
# ==========================================

# Difference between course over ground and heading
df["heading_cog_diff"] = abs(df["cog"] - df["heading"])

# Circular correction:
# e.g. COG=359 and heading=1 should have difference 2,
# not 358.
df["heading_cog_diff"] = df["heading_cog_diff"].apply(
    lambda x: min(x, 360 - x) if pd.notna(x) else np.nan
)


# Speed categories
df["is_stationary"] = (df["sog"] < 0.5).astype(int)

df["is_slow"] = (
    (df["sog"] >= 0.5) &
    (df["sog"] < 5)
).astype(int)

df["is_moving"] = (df["sog"] >= 5).astype(int)


# ==========================================
# SELECT ML FEATURES
# ==========================================

features = [
    "sog",
    "cog",
    "heading",
    "heading_cog_diff",
    "length",
    "width",
    "draught"
]

X = df[features].copy()


# ==========================================
# HANDLE MISSING VALUES
# ==========================================

X = X.fillna(X.median())


print("\nFeature dataset:")
print(X.head())

print("\nFeature shape:", X.shape)

print(X.head())
print(X.shape)