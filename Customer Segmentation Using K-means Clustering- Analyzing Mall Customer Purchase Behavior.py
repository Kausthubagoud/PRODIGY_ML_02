import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('/Users/virinchisai/Downloads/PROJECTS/Prodigy Infotech/PRODIGY_ML_02/Mall_Customers.csv')

# Preprocess the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.iloc[:, 2:])

# Determine the optimal number of clusters using the Elbow method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot the Elbow curve and the customer segmentation scatter plot side by side
fig, axs = plt.subplots(1, 2, figsize=(18, 6))

# Elbow curve
axs[0].plot(range(1, 11), inertia, marker='o')
axs[0].set_xlabel('Number of Clusters')
axs[0].set_ylabel('Inertia')
axs[0].set_title('Elbow Method')

# K-means clustering and customer segmentation
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Add the cluster labels to the original dataset
data['Cluster'] = clusters

# Analyze the clusters
cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=data.columns[2:-1])
cluster_centers['Cluster'] = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']

# Customer segmentation scatter plot
for cluster in data['Cluster'].unique():
    axs[1].scatter(data[data['Cluster'] == cluster]['Annual Income (k$)'],
                   data[data['Cluster'] == cluster]['Spending Score (1-100)'],
                   label=f'Cluster {cluster}')
axs[1].scatter(cluster_centers['Annual Income (k$)'], cluster_centers['Spending Score (1-100)'],
               marker='X', c='black', label='Centroid')
axs[1].set_xlabel('Annual Income (k$)')
axs[1].set_ylabel('Spending Score (1-100)')
axs[1].set_title('Customer Segmentation')
axs[1].legend()

plt.savefig('customer_segmentation.png')
plt.show()
