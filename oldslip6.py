import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

# Load the dataset
data = pd.read_csv('student_data.csv')
# data = pd.read_csv('Mall_Customers.csv')

# url = 'https://docs.google.com/spreadsheets/d/1OERD4rpun1BV2CazUZ22bHXF_CR3KGQa69mRmgV9rMI/edit?usp=sharing'
# data = pd.read_csv(url)

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head())

# Assuming the dataset has numerical features for clustering (e.g., 'Annual Income' and 'Spending Score')
X = data.iloc[:, [3, 4]].values  # Adjust column indices based on your dataset

# **Dendrogram to find the optimal number of clusters**
plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

# **Fitting Agglomerative Clustering**
# Based on the dendrogram, choose the optimal number of clusters (e.g., 5)
n_clusters = 5
hc = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

# **Visualizing the Clusters**
plt.figure(figsize=(10, 7))
for i in range(n_clusters):
    plt.scatter(X[y_hc == i, 0], X[y_hc == i, 1], label=f'Cluster {i+1}')
plt.title('Clusters of Customers')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()
