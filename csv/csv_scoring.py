import pandas as pd

file_path = "csv/2017_dataset_with_amenities.csv"
df = pd.read_csv(file_path)

weights = {
    "MRT_300m": 5,
    "MRT_500m": 4,
    "MRT_1000m": 3,
    "Bus_300m": 3,
    "Bus_500m": 2,
    "Hospitals_300m": 5,
    "Hospitals_500m": 4,
    "Polyclinics_300m": 4,
    "Polyclinics_500m": 3,
    "Malls_300m": 3,
    "Malls_500m": 2,
    "Floor_Area": 2,
    "Lease_Remaining": 3
}

df['Score'] = (
    df['MrtStns_within_300m'].notna().astype(int) * weights['MRT_300m'] +
    df['MrtStns_within_500m'].notna().astype(int) * weights['MRT_500m'] +
    df['MrtStns_within_1000m'].notna().astype(int) * weights['MRT_1000m'] +
    df['busStops_within_300m'].notna().astype(int) * weights['Bus_300m'] +
    df['busStops_within_500m'].notna().astype(int) * weights['Bus_500m'] +
    df['hospitals_within_300m'].notna().astype(int) * weights['Hospitals_300m'] +
    df['hospitals_within_500m'].notna().astype(int) * weights['Hospitals_500m'] +
    df['polyclinics_within_300m'].notna().astype(int) * weights['Polyclinics_300m'] +
    df['polyclinics_within_500m'].notna().astype(int) * weights['Polyclinics_500m'] +
    df['malls_within_300m'].notna().astype(int) * weights['Malls_300m'] +
    df['malls_within_500m'].notna().astype(int) * weights['Malls_500m'] +
    (df['floor_area_sqm'] / df['floor_area_sqm'].max()) * weights['Floor_Area'] * 10 +
    (df['lease_commence_date'] / df['lease_commence_date'].max()) * weights['Lease_Remaining'] * 10
)

# Save to CSV
df.to_csv("flats_with_scores.csv", index=False)
