import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load dataset
df = pd.read_csv("../2017 - 2023.csv")  # Update with the correct path
df = df.drop(columns=["price_per_square_meter", "block", "street_name", "latitude", "longitude"], errors="ignore")

# Select only numerical features for correlation analysis
numerical_df = df.select_dtypes(include=[np.number])
no_target_df = numerical_df.drop(columns=["resale_price"], errors="ignore")

# Compute Variance Inflation Factor (VIF)
vif_data = pd.DataFrame()
vif_data["Feature"] = no_target_df.columns
vif_data["VIF Score"] = [variance_inflation_factor(no_target_df.values, i) for i in range(no_target_df.shape[1])]

# Save VIF data to CSV
vif_csv_path = "vif_results.csv"
vif_data.to_csv(vif_csv_path, index=False)

# Compute correlation matrices
corr_matrix = numerical_df.corr()
spearman_corr_matrix = numerical_df.corr(method='spearman')

# Plot and save Pearson correlation heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.xticks(rotation=45, ha="right")
plt.title("Pearson Correlation Matrix")
pearson_img_path = "pearson_correlation_matrix.png"
plt.savefig(pearson_img_path, bbox_inches="tight")
plt.close()

# Plot and save Spearman correlation heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.xticks(rotation=45, ha="right")
plt.title("Spearman Correlation Matrix")
spearman_img_path = "spearman_correlation_matrix.png"
plt.savefig(spearman_img_path, bbox_inches="tight")
plt.close()






