import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

# Load dataset
data = pd.read_csv('student_data.csv')

print("Dataset Preview:")
print(data.head())

# Select only NUMERIC columns (exclude names, grades, result, etc.)
# Based on your output, numeric columns likely include: Python, and other subject scores, Total, BACKLOG
numeric_cols = data.select_dtypes(include=[np.number]).columns
print(f"\nNumeric columns found: {list(numeric_cols)}")

# Remove enrollment number and other ID columns if present
cols_to_exclude = ['Enrollnment no', 'BACKLOG']  # Adjust based on your needs
X = data[numeric_cols].drop(columns=cols_to_exclude, errors='ignore')

print(f"\nColumns used for clustering: {list(X.columns)}")
print(f"Shape: {X.shape}")

# Handle missing values if any
X = X.fillna(X.mean())

# Create dendrogram
plt.figure(figsize=(12, 6))
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Student Index')
plt.ylabel('Euclidean Distance')
plt.show()