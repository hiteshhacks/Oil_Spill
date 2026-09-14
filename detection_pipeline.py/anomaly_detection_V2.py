import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ==========================================
# LOAD
# ==========================================

df = pd.read_csv("DATA/matched_ais_clean.csv")

tanker_df = df[
    df["shiptype"] == "Tanker"
].copy()


# ==========================================
# ANGULAR FEATURES
# ==========================================

tanker_df["cog_sin"] = np.sin(
    np.radians(tanker_df["cog"])
)

tanker_df["cog_cos"] = np.cos(
    np.radians(tanker_df["cog"])
)

tanker_df["heading_sin"] = np.sin(
    np.radians(tanker_df["heading"])
)

tanker_df["heading_cos"] = np.cos(
    np.radians(tanker_df["heading"])
)


# Angular difference
tanker_df["heading_cog_diff"] = abs(
    tanker_df["cog"] -
    tanker_df["heading"]
)

tanker_df["heading_cog_diff"] = tanker_df[
    "heading_cog_diff"
].apply(
    lambda x: min(x, 360 - x)
    if pd.notna(x) else np.nan
)


# ==========================================
# FEATURES
# ==========================================

features = [
    "sog",
    "cog_sin",
    "cog_cos",
    "heading_sin",
    "heading_cos",
    "heading_cog_diff"
]

X = tanker_df[features].copy()

X = X.fillna(X.median())


# ==========================================
# SCALE
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ==========================================
# ISOLATION FOREST
# ==========================================

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42,
    n_jobs=-1
)

model.fit(X_scaled)


# ==========================================
# PREDICTION
# ==========================================

tanker_df["anomaly"] = model.predict(X_scaled)

tanker_df["is_anomaly"] = (
    tanker_df["anomaly"] == -1
).astype(int)

tanker_df["anomaly_score"] = (
    -model.decision_function(X_scaled)
)


# ==========================================
# RESULTS
# ==========================================

print("\n========== TANKER V2 ==========")

print("Tanker records:", len(tanker_df))

print(
    "Anomalous records:",
    tanker_df["is_anomaly"].sum()
)

print(
    "Anomaly rate:",
    round(
        tanker_df["is_anomaly"].mean() * 100,
        2
    ),
    "%"
)


# print("\n========== TOP ANOMALIES ==========\n")

# print(
#     tanker_df.sort_values(
#         "anomaly_score",
#         ascending=False
#     )[
#         [
#             "mmsi",
#             "sog",
#             "cog",
#             "heading",
#             "heading_cog_diff",
#             "length",
#             "width",
#             "draught",
#             "anomaly_score"
#         ]
#     ].head(20).to_string(index=False)
# )

print("\n========== NAVIGATIONAL STATUS ==========\n")

print(
    tanker_df["navigationalstatus"]
    .value_counts(dropna=False)
)


print("\n========== SOG DISTRIBUTION ==========\n")

print(tanker_df["sog"].describe())

print("\nStationary tankers:")
print(
    (tanker_df["sog"] < 0.5).sum()
)

print(
    "Stationary percentage:",
    round(
        (tanker_df["sog"] < 0.5).mean() * 100,
        2
    ),
    "%"
)


print("\n========== ANOMALY BY SOG STATE ==========\n")

tanker_df["movement_state"] = np.where(
    tanker_df["sog"] < 0.5,
    "Stationary",
    "Moving"
)

print(
    tanker_df.groupby("movement_state")[
        "is_anomaly"
    ].agg(["count", "sum"])
)


# ==========================================
# MOVING TANKER ANOMALY ANALYSIS
# ==========================================

moving = tanker_df[
    tanker_df["sog"] >= 0.5
].copy()

normal = moving[
    moving["is_anomaly"] == 0
]

anomalous = moving[
    moving["is_anomaly"] == 1
]


print("\n========== MOVING TANKER ANALYSIS ==========")

print("\nNormal moving tankers:")
print(
    normal["heading_cog_diff"].describe()
)

print("\nAnomalous moving tankers:")
print(
    anomalous["heading_cog_diff"].describe()
)


# Compare averages

print("\n========== MEAN COMPARISON ==========")

print(
    "Normal mean SOG:",
    round(normal["sog"].mean(), 2)
)

print(
    "Anomaly mean SOG:",
    round(anomalous["sog"].mean(), 2)
)

print(
    "Normal mean heading/COG difference:",
    round(
        normal["heading_cog_diff"].mean(),
        2
    )
)

print(
    "Anomaly mean heading/COG difference:",
    round(
        anomalous["heading_cog_diff"].mean(),
        2
    )
)


# Percentage of anomalies with large angular difference

for threshold in [30, 60, 90, 120, 150]:

    percentage = (
        (anomalous["heading_cog_diff"] > threshold)
        .mean()
        * 100
    )

    print(
        f"Anomalies with difference > {threshold}°:",
        round(percentage, 2),
        "%"
    )