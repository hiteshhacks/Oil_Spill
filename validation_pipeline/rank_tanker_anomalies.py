import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


# ============================================================
# 1. Load cleaned matched AIS data
# ============================================================

df = pd.read_csv("DATA/matched_ais_clean.csv")

print("Total AIS records:", len(df))


# ============================================================
# 2. Keep only Tankers
# ============================================================

tanker = df[df["shiptype"] == "Tanker"].copy()

print("Tanker records:", len(tanker))
print("Unique tankers:", tanker["mmsi"].nunique())


# ============================================================
# 3. Feature engineering
# ============================================================

# Circular representation of COG
tanker["cog_sin"] = np.sin(np.radians(tanker["cog"]))
tanker["cog_cos"] = np.cos(np.radians(tanker["cog"]))

# Circular representation of Heading
tanker["heading_sin"] = np.sin(np.radians(tanker["heading"]))
tanker["heading_cos"] = np.cos(np.radians(tanker["heading"]))


# Absolute circular difference between heading and COG
diff = abs(tanker["heading"] - tanker["cog"])

tanker["heading_cog_diff"] = np.minimum(
    diff,
    360 - diff
)


# ============================================================
# 4. Select ML features
# ============================================================

features = [
    "sog",
    "cog_sin",
    "cog_cos",
    "heading_sin",
    "heading_cos",
    "heading_cog_diff"
]

X = tanker[features].copy()


# ============================================================
# 5. Handle missing values
# ============================================================

X = X.fillna(X.median())


# ============================================================
# 6. Standardize features
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ============================================================
# 7. Isolation Forest
# ============================================================

model = IsolationForest(
    n_estimators=200,
    contamination=0.02,
    random_state=42
)

model.fit(X_scaled)


# ============================================================
# 8. Generate predictions
# ============================================================

tanker["prediction"] = model.predict(X_scaled)

# -1 = anomaly
# +1 = normal

tanker["is_anomaly"] = tanker["prediction"] == -1


# Isolation Forest:
# higher score = more normal
# convert it so higher = more anomalous

tanker["anomaly_score"] = -model.decision_function(X_scaled)


# ============================================================
# 9. Extract anomalous observations
# ============================================================

anomalies = tanker[tanker["is_anomaly"]].copy()

print("\n========================================")
print("AIS ANOMALY RESULTS")
print("========================================")

print("Total tanker observations :", len(tanker))
print("Anomalous observations     :", len(anomalies))
print("Anomaly percentage         :",
      round(len(anomalies) / len(tanker) * 100, 3))


# ============================================================
# 10. Save all anomalous observations
# ============================================================

anomalies.to_csv(
    "tanker_anomaly_v2_02.csv",
    index=False
)

print("\nSaved:")
print("tanker_anomaly_v2_02.csv")


# ============================================================
# 11. Aggregate anomalies by vessel
# ============================================================

ranking = (
    anomalies
    .groupby("mmsi")
    .agg(
        anomaly_records=("mmsi", "size"),

        max_anomaly_score=("anomaly_score", "max"),

        mean_anomaly_score=("anomaly_score", "mean"),

        max_heading_cog_diff=("heading_cog_diff", "max"),

        mean_heading_cog_diff=("heading_cog_diff", "mean"),

        mean_sog=("sog", "mean"),

        max_sog=("sog", "max"),

        length=("length", "first"),

        width=("width", "first"),

        draught=("draught", "first")
    )
    .reset_index()
)


# ============================================================
# 12. Calculate a simple ranking score
# ============================================================

# Normalize individual indicators

def normalize(series):
    if series.max() == series.min():
        return pd.Series(0.0, index=series.index)

    return (
        (series - series.min()) /
        (series.max() - series.min())
    )


ranking["score_anomaly"] = normalize(
    ranking["max_anomaly_score"]
)

ranking["score_frequency"] = normalize(
    ranking["anomaly_records"]
)

ranking["score_heading"] = normalize(
    ranking["max_heading_cog_diff"]
)


# Weighted vessel-level suspiciousness score
ranking["vessel_risk_score"] = (
    0.50 * ranking["score_anomaly"] +
    0.30 * ranking["score_frequency"] +
    0.20 * ranking["score_heading"]
)


# ============================================================
# 13. Rank vessels
# ============================================================

ranking = ranking.sort_values(
    "vessel_risk_score",
    ascending=False
).reset_index(drop=True)

ranking["rank"] = ranking.index + 1


# ============================================================
# 14. Reorder columns
# ============================================================

ranking = ranking[
    [
        "rank",
        "mmsi",
        "anomaly_records",
        "max_anomaly_score",
        "mean_anomaly_score",
        "max_heading_cog_diff",
        "mean_heading_cog_diff",
        "mean_sog",
        "max_sog",
        "length",
        "width",
        "draught",
        "vessel_risk_score"
    ]
]


# ============================================================
# 15. Save ranking
# ============================================================

ranking.to_csv(
    "tanker_suspicious_vessels.csv",
    index=False
)

print("\nSaved:")
print("tanker_suspicious_vessels.csv")


# ============================================================
# 16. Display Top 20
# ============================================================

print("\n========================================")
print("TOP 20 SUSPICIOUS TANKERS")
print("========================================")

print(
    ranking.head(20).to_string(index=False)
)