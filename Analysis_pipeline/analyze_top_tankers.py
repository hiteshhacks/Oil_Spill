import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Load data
# ============================================================

ais = pd.read_csv("DATA/matched_ais_clean.csv")
ranking = pd.read_csv("DATA/tanker_suspicious_vessels.csv")


# ============================================================
# 2. Select Top 10 tankers
# ============================================================

top10 = ranking.head(10)["mmsi"].tolist()

print("Top 10 MMSIs:")
for i, mmsi in enumerate(top10, 1):
    print(f"{i}. {mmsi}")


# ============================================================
# 3. Extract Top-10 tanker observations
# ============================================================

data = ais[
    (ais["mmsi"].isin(top10)) &
    (ais["shiptype"] == "Tanker")
].copy()


# ============================================================
# 4. Feature engineering
# ============================================================

diff = abs(data["heading"] - data["cog"])

data["heading_cog_diff"] = np.minimum(
    diff,
    360 - diff
)


# ============================================================
# 5. Recreate Isolation Forest
# ============================================================

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


data["cog_sin"] = np.sin(np.radians(data["cog"]))
data["cog_cos"] = np.cos(np.radians(data["cog"]))

data["heading_sin"] = np.sin(np.radians(data["heading"]))
data["heading_cos"] = np.cos(np.radians(data["heading"]))


features = [
    "sog",
    "cog_sin",
    "cog_cos",
    "heading_sin",
    "heading_cos",
    "heading_cog_diff"
]

X = data[features].copy()
X = X.fillna(X.median())


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


model = IsolationForest(
    n_estimators=200,
    contamination=0.02,
    random_state=42
)

model.fit(X_scaled)


data["prediction"] = model.predict(X_scaled)

data["is_anomaly"] = data["prediction"] == -1

data["anomaly_score"] = -model.decision_function(X_scaled)


# ============================================================
# 6. Save Top-10 detailed observations
# ============================================================

data.to_csv(
    "top10_tanker_analysis.csv",
    index=False
)

print("\nSaved: top10_tanker_analysis.csv")


# ============================================================
# 7. Summary table
# ============================================================

summary = (
    data
    .groupby("mmsi")
    .agg(
        total_records=("mmsi", "size"),
        anomaly_records=("is_anomaly", "sum"),
        anomaly_rate=("is_anomaly", "mean"),
        max_anomaly_score=("anomaly_score", "max"),
        mean_anomaly_score=("anomaly_score", "mean"),
        max_heading_cog_diff=("heading_cog_diff", "max"),
        mean_heading_cog_diff=("heading_cog_diff", "mean"),
        mean_sog=("sog", "mean"),
        max_sog=("sog", "max")
    )
    .reset_index()
)

summary["anomaly_rate"] *= 100

summary = summary.sort_values(
    "anomaly_records",
    ascending=False
)


print("\n========================================")
print("TOP 10 TANKER BEHAVIOR SUMMARY")
print("========================================")

print(
    summary.to_string(
        index=False,
        formatters={
            "anomaly_rate": "{:.2f}%".format,
            "max_anomaly_score": "{:.4f}".format,
            "mean_anomaly_score": "{:.4f}".format,
            "max_heading_cog_diff": "{:.1f}".format,
            "mean_heading_cog_diff": "{:.1f}".format,
            "mean_sog": "{:.2f}".format,
            "max_sog": "{:.1f}".format
        }
    )
)

summary.to_csv(
    "top10_tanker_behavior_summary.csv",
    index=False
)


# ============================================================
# 8. Plot 1 — Anomaly records by vessel
# ============================================================

plt.figure(figsize=(12, 6))

plt.bar(
    summary["mmsi"].astype(str),
    summary["anomaly_records"]
)

plt.xlabel("MMSI")
plt.ylabel("Number of anomalous observations")
plt.title("Anomalous AIS Observations for Top 10 Tankers")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "top10_anomaly_frequency.png",
    dpi=300
)

plt.show()


# ============================================================
# 9. Plot 2 — Maximum anomaly score
# ============================================================

score_sorted = summary.sort_values(
    "max_anomaly_score",
    ascending=False
)

plt.figure(figsize=(12, 6))

plt.bar(
    score_sorted["mmsi"].astype(str),
    score_sorted["max_anomaly_score"]
)

plt.xlabel("MMSI")
plt.ylabel("Maximum anomaly score")
plt.title("Maximum AIS Anomaly Score by Tanker")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "top10_max_anomaly_score.png",
    dpi=300
)

plt.show()


# ============================================================
# 10. Plot 3 — Heading vs COG discrepancy
# ============================================================

diff_sorted = summary.sort_values(
    "max_heading_cog_diff",
    ascending=False
)

plt.figure(figsize=(12, 6))

plt.bar(
    diff_sorted["mmsi"].astype(str),
    diff_sorted["max_heading_cog_diff"]
)

plt.xlabel("MMSI")
plt.ylabel("Maximum |Heading − COG| (degrees)")
plt.title("Maximum Heading–COG Discrepancy for Top 10 Tankers")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "top10_heading_cog_difference.png",
    dpi=300
)

plt.show()


# ============================================================
# 11. Plot 4 — SOG vs anomaly score
# ============================================================

plt.figure(figsize=(10, 7))

normal = data[~data["is_anomaly"]]
anomaly = data[data["is_anomaly"]]

plt.scatter(
    normal["sog"],
    normal["anomaly_score"],
    alpha=0.25,
    label="Normal"
)

plt.scatter(
    anomaly["sog"],
    anomaly["anomaly_score"],
    alpha=0.8,
    label="Anomaly"
)

plt.xlabel("Speed Over Ground (knots)")
plt.ylabel("Anomaly Score")
plt.title("AIS Anomaly Score vs Speed Over Ground")

plt.legend()

plt.tight_layout()

plt.savefig(
    "sog_vs_anomaly_score.png",
    dpi=300
)

plt.show()


# ============================================================
# 12. Plot 5 — Heading-COG difference vs anomaly score
# ============================================================

plt.figure(figsize=(10, 7))

plt.scatter(
    normal["heading_cog_diff"],
    normal["anomaly_score"],
    alpha=0.25,
    label="Normal"
)

plt.scatter(
    anomaly["heading_cog_diff"],
    anomaly["anomaly_score"],
    alpha=0.8,
    label="Anomaly"
)

plt.xlabel("Heading–COG Difference (degrees)")
plt.ylabel("Anomaly Score")
plt.title("AIS Anomaly Score vs Heading–COG Difference")

plt.legend()

plt.tight_layout()

plt.savefig(
    "heading_cog_vs_anomaly_score.png",
    dpi=300
)

plt.show()


# ============================================================
# 13. Print actual anomalous observations
# ============================================================

print("\n========================================")
print("TOP ANOMALOUS OBSERVATIONS")
print("========================================")

anomaly_table = (
    data[data["is_anomaly"]]
    .sort_values(
        "anomaly_score",
        ascending=False
    )
    [
        [
            "mmsi",
            "sog",
            "cog",
            "heading",
            "heading_cog_diff",
            "anomaly_score"
        ]
    ]
)

print(
    anomaly_table.head(30).to_string(
        index=False
    )
)