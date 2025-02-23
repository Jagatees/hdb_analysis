import pandas as pd

unique_df = pd.read_csv('2017_dataset_lon_lat.csv')
print(unique_df.shape[0])

# Filter rows where both latitude and longitude are 0.0
zero_coordinates = unique_df[(unique_df['latitude'] == 0.0) & (unique_df['longitude'] == 0.0)]

# Display the filtered rows
print(len(zero_coordinates))

unique_df = unique_df[~((unique_df['latitude'] == 0.0) & (unique_df['longitude'] == 0.0))]
# Verify the rows are deleted
print(unique_df.shape[0])

# Save the DataFrame to an Excel file
output_file = '2017_dataset.csv'
unique_df.to_csv(output_file, index=False)
print(f"DataFrame successfully saved to {output_file}")