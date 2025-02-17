import csv
from math import radians, sin, cos, sqrt, atan2
import pandas as pd

dataset = pd.read_csv("csv/2017_dataset_lon_lat.csv")

hospitals = pd.read_csv("csv/hospitals_with_coordinates.csv")
malls = pd.read_csv("csv/shopping_mall_coordinates.csv")
polyclinics = pd.read_csv("csv/Polyclinics_dataset.csv")
bus_stops = pd.read_csv("csv/bus_stops_coordinates.csv")
mrt_stations = pd.read_csv("csv/MRT Stations.csv")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Radius of the Earth in meters
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c  # Distance in meters

distance_thresholds = [300, 500, 1000, 2000]

for d in distance_thresholds:
    dataset[f'hospitals_within_{d}m'] = ""
    dataset[f'malls_within_{d}m'] = ""
    dataset[f'polyclinics_within_{d}m'] = ""
    dataset[f'busStops_within_{d}m'] = ""
    dataset[f'MrtStns_within_{d}m'] = ""
    
for i, row in dataset.iterrows():
    print(i)
    lat,lon = row['latitude'], row['longitude']    
    
    hospital_names = {d: [] for d in distance_thresholds}
    for _, hosp in hospitals.iterrows():
        dist = haversine(lat, lon, hosp['latitude'], hosp['longitude'])
        for d in distance_thresholds:
            if dist <= d:
                hospital_names[d].append(hosp['hospital_name']) 
                
    mall_names = {d: [] for d in distance_thresholds}
    for _, mall in malls.iterrows():
        dist = haversine(lat, lon, mall['LATITUDE'], mall['LONGITUDE'])
        for d in distance_thresholds:
            if dist <= d:
                mall_names[d].append(mall['Mall Name'])             
                
    polyclinic_names = {d: [] for d in distance_thresholds}
    for _, poly in polyclinics.iterrows():
        dist = haversine(lat, lon, poly['latitude'], poly['longitude'])
        for d in distance_thresholds:
            if dist <= d:
                polyclinic_names[d].append(poly['Name'])             
                
    busStop_names = {d: [] for d in distance_thresholds}
    for _, bus in bus_stops.iterrows():
        dist = haversine(lat, lon, bus['Latitude'], bus['Longitude'])
        for d in distance_thresholds:
            if dist <= d:
                busStop_names[d].append(str(bus['BusStopCode']))            
                
    mrtStns = {d: [] for d in distance_thresholds}
    for _, mrt in mrt_stations.iterrows():
        dist = haversine(lat, lon, mrt['Latitude'], mrt['Longitude'])
        for d in distance_thresholds:
            if dist <= d:
                mrtStns[d].append(mrt['STN_NAME'])             
                

    for d in distance_thresholds:
        dataset.at[i, f'hospitals_within_{d}m'] = ", ".join(hospital_names[d])
        dataset.at[i, f'malls_within_{d}m'] = ", ".join(mall_names[d])
        dataset.at[i, f'polyclinics_within_{d}m'] = ", ".join(polyclinic_names[d])
        dataset.at[i, f'busStops_within_{d}m'] = ", ".join(busStop_names[d])
        dataset.at[i, f'MrtStns_within_{d}m'] = ", ".join(mrtStns[d])
        
 
output_path = "csv/2017_dataset_with_amenities.csv"
dataset.to_csv(output_path, index=False)
output_path