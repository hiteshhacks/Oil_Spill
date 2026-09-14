import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ==========================================
# 1. LOAD TANKER DATA
# ==========================================

df = pd.read_csv("DATA/matched_ais_clean.csv")

df = df[
    df["shiptype"] == "Tanker"
].copy()


# ==========================================
# 2. ANGULAR FEATURES
# ==========================================

df["cog_sin"] = np.sin(
    np.radians(df["cog"])
)

df["cog_cos"] = np.cos(
    np.radians(df["cog"])
)

df["heading_sin"] = np.sin(
    np.radians(df["heading"])
)

df["heading_cos"] = np.cos(
    np.radians(df["heading"])
)


df["heading_cog_diff"] = abs(
    df["cog"] - df["heading"]
)

df["heading_cog_diff"] = (
    df["heading_cog_diff"]
    .apply(
        lambda x: min(x, 360 - x)
        if pd.notna(x) else np.nan
    )
)


# ==========================================
# 3. VALIDATION RULE
# ==========================================

df["is_moving"] = (
    df["sog"] >= 0.5
).astype(int)


df["rule_anomaly"] = (
    (df["is_moving"] == 1) &
    (df["heading_cog_diff"] > 30)
).astype(int)


# ==========================================
# 4. FEATURES
# ==========================================

features = [
    "sog",
    "cog_sin",
    "cog_cos",
    "heading_sin",
    "heading_cos",
    "heading_cog_diff"
]

X = df[features].copy()

X = X.fillna(X.median())


# ==========================================
# 5. SCALE
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ==========================================
# 6. CONTAMINATION EXPERIMENT
# ==========================================

contamination_values = [
    0.02,
    0.05,
    0.10
]

results = []


for contamination in contamination_values:

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_scaled)

    predictions = model.predict(X_scaled)

    df["prediction"] = (
        predictions == -1
    ).astype(int)


    # --------------------------------------
    # Number of anomalies
    # --------------------------------------

    total_anomalies = df[
        "prediction"
    ].sum()


    # --------------------------------------
    # Rule anomaly detection
    # --------------------------------------

    rule_records = df[
        df["rule_anomaly"] == 1
    ]

    detected_rule = rule_records[
        rule_records["prediction"] == 1
    ]


    coverage = (
        len(detected_rule) /
        len(rule_records) * 100
    )


    # --------------------------------------
    # Results
    # --------------------------------------

    results.append({

        "contamination": contamination,

        "total_anomalies":
            total_anomalies,

        "anomaly_rate":
            total_anomalies /
            len(df) * 100,

        "rule_anomalies":
            len(rule_records),

        "detected_rule_anomalies":
            len(detected_rule),

        "coverage":
            coverage
    })


# ==========================================
# 7. DISPLAY RESULTS
# ==========================================

results_df = pd.DataFrame(results)

print("\n========== CONTAMINATION EXPERIMENT ==========\n")

print(
    results_df.to_string(
        index=False
    )
)


# ==========================================
# 8. SAVE RESULTS
# ==========================================

results_df.to_csv(
    "contamination_results.csv",
    index=False
)

print(
    "\nSaved: contamination_results.csv"
)