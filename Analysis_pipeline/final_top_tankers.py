import pandas as pd

# Load the original 174 anomalies
anomalies = pd.read_csv("DATA/tanker_anomaly_v2_02.csv")

# Load original vessel ranking
ranking = pd.read_csv("DATA/tanker_suspicious_vessels.csv")

# Top 10 vessels from the ORIGINAL ranking
top10 = ranking.head(10)["mmsi"].tolist()

# Keep only anomaly observations belonging to Top 10
top10_anomalies = anomalies[
    anomalies["mmsi"].isin(top10)
].copy()

print("Top-10 anomalous observations:",
      len(top10_anomalies))


# ------------------------------------------------------------
# Aggregate using the ORIGINAL anomaly results
# ------------------------------------------------------------

summary = (
    top10_anomalies
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


# Merge total observations from original tanker dataset
ais = pd.read_csv("DATA/matched_ais_clean.csv")

tanker = ais[
    ais["shiptype"] == "Tanker"
]

total_records = (
    tanker
    .groupby("mmsi")
    .size()
    .reset_index(name="total_records")
)

summary = summary.merge(
    total_records,
    on="mmsi",
    how="left"
)

summary["anomaly_rate"] = (
    summary["anomaly_records"] /
    summary["total_records"] *
    100
)


# Preserve original ranking
summary = summary.merge(
    ranking[
        ["mmsi", "rank", "vessel_risk_score"]
    ],
    on="mmsi",
    how="left"
)

summary = summary.sort_values(
    "rank"
)


# Reorder
summary = summary[
    [
        "rank",
        "mmsi",
        "total_records",
        "anomaly_records",
        "anomaly_rate",
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


print("\n========================================")
print("FINAL TOP-10 ANALYSIS")
print("========================================")

print(
    summary.to_string(index=False)
)


# summary.to_csv(
#     "FINAL_top10_tanker_analysis.csv",
#     index=False
# )

# top10_anomalies.to_csv(
#     "FINAL_top10_anomalous_observations.csv",
#     index=False
# )