import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("DATA/matched_ais_clean.csv")

print("Records:", len(df))
print("Vessels:", df["mmsi"].nunique())


# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================

df["heading_cog_diff"] = abs(df["cog"] - df["heading"])

# Circular angle correction
df["heading_cog_diff"] = df["heading_cog_diff"].apply(
    lambda x: min(x, 360 - x)
    if pd.notna(x) else np.nan
)


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
# 3. HANDLE MISSING VALUES
# ==========================================

X = X.fillna(X.median())


# ==========================================
# 4. SCALE FEATURES
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ==========================================
# 5. ISOLATION FOREST
# ==========================================

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42,
    n_jobs=-1
)

model.fit(X_scaled)


# ==========================================
# 6. PREDICT ANOMALIES
# ==========================================

# 1  = normal
# -1 = anomaly

df["anomaly"] = model.predict(X_scaled)


# Convert to easier interpretation
df["is_anomaly"] = (
    df["anomaly"] == -1
).astype(int)


# Anomaly score
df["anomaly_score"] = -model.decision_function(X_scaled)


# ==========================================
# 7. RESULTS
# ==========================================

total = len(df)

anomalies = df["is_anomaly"].sum()

normal = total - anomalies

print("\n========== ANOMALY RESULTS ==========")

print("Total records:", total)
print("Normal records:", normal)
print("Anomalous records:", anomalies)

print(
    "Anomaly percentage:",
    round(anomalies / total * 100, 2),
    "%"
)


# ==========================================
# 8. TOP ANOMALOUS RECORDS
# ==========================================

top_anomalies = df.sort_values(
    "anomaly_score",
    ascending=False
)

print("\n========== TOP ANOMALIES ==========\n")

print(
    top_anomalies[
        [
            "mmsi",
            "sog",
            "cog",
            "heading",
            "heading_cog_diff",
            "shiptype",
            "length",
            "width",
            "draught",
            "anomaly_score"
        ]
    ].head(20).to_string(index=False)
)


# ==========================================
# 9. ANOMALIES BY SHIP TYPE
# ==========================================

print("\n========== ANOMALIES BY SHIP TYPE ==========\n")

anomaly_by_type = (
    df.groupby("shiptype")["is_anomaly"]
    .agg(["count", "sum"])
)

anomaly_by_type["anomaly_rate"] = (
    anomaly_by_type["sum"]
    / anomaly_by_type["count"]
    * 100
)

print(
    anomaly_by_type.sort_values(
        "anomaly_rate",
        ascending=False
    )
)


# # ==========================================
# # 10. SAVE RESULTS
# # ==========================================

# df.to_csv(
#     "ais_anomaly_results.csv",
#     index=False
# )

# print("\nSaved: ais_anomaly_results.csv")


# ==========================================
# TANKER ANOMALY ANALYSIS
# ==========================================

tanker_df = df[
    df["shiptype"] == "Tanker"
].copy()

print("Tanker records:", len(tanker_df))
print("Tanker vessels:", tanker_df["mmsi"].nunique())


# ------------------------------------------
# Features
# ------------------------------------------

tanker_features = [
    "sog",
    "cog",
    "heading",
    "heading_cog_diff",
    "length",
    "width",
    "draught"
]

X_tanker = tanker_df[tanker_features].copy()

X_tanker = X_tanker.fillna(
    X_tanker.median()
)


# ------------------------------------------
# Scaling
# ------------------------------------------

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_tanker_scaled = scaler.fit_transform(X_tanker)


# ------------------------------------------
# Isolation Forest
# ------------------------------------------

from sklearn.ensemble import IsolationForest

tanker_model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42,
    n_jobs=-1
)

tanker_model.fit(X_tanker_scaled)


# ------------------------------------------
# Prediction
# ------------------------------------------

tanker_df["anomaly"] = tanker_model.predict(
    X_tanker_scaled
)

tanker_df["is_anomaly"] = (
    tanker_df["anomaly"] == -1
).astype(int)

tanker_df["anomaly_score"] = (
    -tanker_model.decision_function(
        X_tanker_scaled
    )
)


# ------------------------------------------
# Results
# ------------------------------------------

print("\n========== TANKER RESULTS ==========")

print(
    "Anomalous tanker records:",
    tanker_df["is_anomaly"].sum()
)

print(
    "Tanker anomaly rate:",
    round(
        tanker_df["is_anomaly"].mean() * 100,
        2
    ),
    "%"
)


# ------------------------------------------
# Top tanker anomalies
# ------------------------------------------

print("\n========== TOP TANKER ANOMALIES ==========\n")

print(
    tanker_df.sort_values(
        "anomaly_score",
        ascending=False
    )[
        [
            "mmsi",
            "sog",
            "cog",
            "heading",
            "heading_cog_diff",
            "length",
            "width",
            "draught",
            "anomaly_score"
        ]
    ].head(20).to_string(index=False)
)