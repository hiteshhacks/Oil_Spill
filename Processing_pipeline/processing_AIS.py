import pandas as pd
import json

# Load JSON
with open("ais_data.json", "r") as f:
    json_data = json.load(f)

json_df = pd.DataFrame(json_data)

# Load CSV
csv_df = pd.read_csv("ais_data.csv")

print("JSON shape:", json_df.shape)
print("CSV shape:", csv_df.shape)



json_df["MMSI"] = json_df["MMSI"].astype(str).str.strip()
csv_df["mmsi"] = csv_df["mmsi"].astype(str).str.strip()

json_mmsi = set(json_df["MMSI"].unique())
csv_mmsi = set(csv_df["mmsi"].unique())

matched_mmsi = json_mmsi.intersection(csv_mmsi)

print("JSON MMSIs:", len(json_mmsi))
print("CSV MMSIs:", len(csv_mmsi))
print("Matched MMSIs:", len(matched_mmsi))


matched_csv = csv_df[csv_df["mmsi"].isin(matched_mmsi)].copy()

print("Matched CSV records:", len(matched_csv))
print("Unique matched vessels:", matched_csv["mmsi"].nunique())


sample_mmsi = list(matched_mmsi)[0]

print("MMSI:", sample_mmsi)

print("\nJSON:")
print(json_df[json_df["MMSI"] == sample_mmsi].T)

print("\nCSV:")
print(matched_csv[matched_csv["mmsi"] == sample_mmsi].head(10))


for col in ["A", "B", "C", "D", "DRAUGHT"]:
    json_df[col] = pd.to_numeric(
        json_df[col],
        errors="coerce"
    )


# JSON vessel dimensions
json_df["length_json"] = (
    json_df["A"] + json_df["B"]
)

json_df["width_json"] = (
    json_df["C"] + json_df["D"]
)

dimension_check = matched_csv.groupby("mmsi").agg({
    "length": "first",
    "width": "first",
    "draught": "first"
}).reset_index()



dimension_check = dimension_check.merge(
    json_df[
        ["MMSI", "length_json", "width_json", "DRAUGHT"]
    ],
    left_on="mmsi",
    right_on="MMSI",
    how="left"
)

# Convert numeric columns to numbers
numeric_cols = [
    "length",
    "width",
    "draught",
    "length_json",
    "width_json",
    "DRAUGHT"
]

for col in numeric_cols:
    dimension_check[col] = pd.to_numeric(
        dimension_check[col],
        errors="coerce"
    )

# Now calculate differences
dimension_check["length_difference"] = (
    dimension_check["length"] -
    dimension_check["length_json"]
).abs()

dimension_check["width_difference"] = (
    dimension_check["width"] -
    dimension_check["width_json"]
).abs()

dimension_check["draught_difference"] = (
    dimension_check["draught"] -
    dimension_check["DRAUGHT"]
).abs()

print(dimension_check.head(20))




# ==========================================
# MATCH QUALITY ANALYSIS
# ==========================================

print("\n========== MATCH QUALITY ==========\n")

print("Total matched vessels:", len(dimension_check))


# Length difference statistics
print("\nLength difference:")
print(dimension_check["length_difference"].describe())


# Width difference statistics
print("\nWidth difference:")
print(dimension_check["width_difference"].describe())


# Draught difference statistics
print("\nDraught difference:")
print(dimension_check["draught_difference"].describe())


# How many have small differences?
print("\nLength difference <= 2m:",
      (dimension_check["length_difference"] <= 2).sum())

print("Width difference <= 2m:",
      (dimension_check["width_difference"] <= 2).sum())

print("Draught difference <= 1m:",
      (dimension_check["draught_difference"] <= 1).sum())


# Show suspicious vessels
suspicious = dimension_check[
    (dimension_check["length_difference"] > 2) |
    (dimension_check["width_difference"] > 2) |
    (dimension_check["draught_difference"] > 2)
]

print("\nSuspicious vessels:", len(suspicious))

print(
    suspicious[
        [
            "mmsi",
            "length",
            "length_json",
            "length_difference",
            "width",
            "width_json",
            "width_difference",
            "draught",
            "DRAUGHT",
            "draught_difference"
        ]
    ].to_string(index=False)
)


# ==========================================
# 11. CLEAN MATCHED AIS DATA
# ==========================================

clean_ais = matched_csv.copy()

print("Original records:", len(clean_ais))


# Remove unnecessary CSV index column if present
if "Unnamed: 0" in clean_ais.columns:
    clean_ais = clean_ais.drop(columns=["Unnamed: 0"])


# ------------------------------------------
# Convert numeric columns
# ------------------------------------------

numeric_columns = [
    "sog",
    "cog",
    "heading",
    "width",
    "length",
    "draught"
]

for col in numeric_columns:
    clean_ais[col] = pd.to_numeric(
        clean_ais[col],
        errors="coerce"
    )


# ------------------------------------------
# Check missing values
# ------------------------------------------

print("\nMissing values:")
print(clean_ais.isnull().sum())


# ------------------------------------------
# Remove impossible SOG values
# ------------------------------------------

clean_ais = clean_ais[
    (clean_ais["sog"] >= 0)
].copy()


# ------------------------------------------
# Normalize COG
# ------------------------------------------

clean_ais.loc[
    (clean_ais["cog"] < 0) |
    (clean_ais["cog"] > 360),
    "cog"
] = pd.NA


# ------------------------------------------
# Normalize heading
# ------------------------------------------

# AIS commonly uses 511 as "not available"
clean_ais.loc[
    clean_ais["heading"] == 511,
    "heading"
] = pd.NA


# ------------------------------------------
# Remove duplicate records
# ------------------------------------------

before = len(clean_ais)

clean_ais = clean_ais.drop_duplicates()

after = len(clean_ais)

print("\nDuplicates removed:", before - after)


# ------------------------------------------
# Final statistics
# ------------------------------------------

print("\n========== CLEAN DATASET ==========")

print("Records:", len(clean_ais))
print("Vessels:", clean_ais["mmsi"].nunique())

print("\nShip types:")
print(clean_ais["shiptype"].value_counts())

print("\nFinal missing values:")
print(clean_ais.isnull().sum())


# ------------------------------------------
# Save
# ------------------------------------------

clean_ais.to_csv(
    "matched_ais_clean.csv",
    index=False
)

print("\nSaved: matched_ais_clean.csv")