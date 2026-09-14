import pandas as pd
import numpy as np


# ==========================================
# 1. LOAD CLEAN AIS DATA
# ==========================================

df = pd.read_csv("DATA/matched_ais_clean.csv")

# We focus on tankers because they are the
# vessel class relevant to our oil-spill study.
df = df[df["shiptype"] == "Tanker"].copy()

print("Tanker records:", len(df))
print("Tanker vessels:", df["mmsi"].nunique())


# ==========================================
# 2. CIRCULAR HEADING-COG DIFFERENCE
# ==========================================

df["heading_cog_diff"] = abs(
    df["cog"] - df["heading"]
)

df["heading_cog_diff"] = df[
    "heading_cog_diff"
].apply(
    lambda x: min(x, 360 - x)
    if pd.notna(x) else np.nan
)


# ==========================================
# 3. MOVEMENT STATE
# ==========================================

# We only evaluate heading/COG consistency
# when the vessel is actually moving.

df["is_moving"] = (
    df["sog"] >= 0.5
).astype(int)


# ==========================================
# 4. DOMAIN-BASED VALIDATION RULES
# ==========================================

# Rule 1:
# Moving vessel with unusually large
# heading/COG disagreement.

rule_heading = (
    (df["is_moving"] == 1) &
    (df["heading_cog_diff"] > 30)
)


# Rule 2:
# Moving vessel with very large
# heading/COG disagreement.

rule_extreme_heading = (
    (df["is_moving"] == 1) &
    (df["heading_cog_diff"] > 60)
)


# Combine rules.
#
# IMPORTANT:
# We use >30° as the primary validation rule.
#
# >60° is retained as a stricter sensitivity
# rule for later analysis.

df["rule_anomaly"] = rule_heading.astype(int)

df["extreme_rule_anomaly"] = (
    rule_extreme_heading
).astype(int)


# ==========================================
# 5. LABEL COUNTS
# ==========================================

print("\n========== VALIDATION LABELS ==========")

print(
    "Rule anomalies (>30°):",
    df["rule_anomaly"].sum()
)

print(
    "Extreme anomalies (>60°):",
    df["extreme_rule_anomaly"].sum()
)

print(
    "Rule anomaly rate:",
    round(
        df["rule_anomaly"].mean() * 100,
        3
    ),
    "%"
)


# ==========================================
# 6. SHOW VALIDATION EXAMPLES
# ==========================================

print("\n========== RULE ANOMALIES ==========\n")

print(
    df[df["rule_anomaly"] == 1][
        [
            "mmsi",
            "sog",
            "cog",
            "heading",
            "heading_cog_diff",
            "navigationalstatus",
            "shiptype"
        ]
    ]
    .sort_values(
        "heading_cog_diff",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)


# ==========================================
# 7. SAVE
# ==========================================

# df.to_csv(
#     "tanker_validation_labels.csv",
#     index=False
# )

# print(
#     "\nSaved: tanker_validation_labels.csv"
# )



# ==========================================
# COMPARE ISOLATION FOREST WITH RULE LABELS
# ==========================================

# Load Isolation Forest results
if_results = pd.read_csv(
    "DATA/ais_anomaly_results.csv"
)

# Keep tanker records only
if_tanker = if_results[
    if_results["shiptype"] == "Tanker"
].copy()


# Create the same heading/COG difference
if_tanker["heading_cog_diff"] = abs(
    if_tanker["cog"] -
    if_tanker["heading"]
)

if_tanker["heading_cog_diff"] = (
    if_tanker["heading_cog_diff"]
    .apply(
        lambda x: min(x, 360 - x)
        if pd.notna(x) else np.nan
    )
)


# Rule-based high-confidence anomalies
if_tanker["rule_anomaly"] = (
    (if_tanker["sog"] >= 0.5) &
    (if_tanker["heading_cog_diff"] > 30)
).astype(int)


# ==========================================
# OVERLAP
# ==========================================

rule_anomalies = if_tanker[
    if_tanker["rule_anomaly"] == 1
]

detected = rule_anomalies[
    rule_anomalies["is_anomaly"] == 1
]


print("\n========== ISOLATION FOREST vs RULE ==========")

print(
    "Rule anomalies:",
    len(rule_anomalies)
)

print(
    "Detected by Isolation Forest:",
    len(detected)
)

print(
    "Detection coverage:",
    round(
        len(detected) /
        len(rule_anomalies) * 100,
        2
    ),
    "%"
)


print("\n========== RULE ANOMALIES ==========\n")

print(
    rule_anomalies[
        [
            "mmsi",
            "sog",
            "cog",
            "heading",
            "heading_cog_diff",
            "is_anomaly",
            "anomaly_score"
        ]
    ]
    .sort_values(
        "heading_cog_diff",
        ascending=False
    )
    .to_string(index=False)
)