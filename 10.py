import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv('breast_cancer_data.csv')
X = data.drop('target', axis=1)
y = data['target']

X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
centers_pca = pca.transform(kmeans.cluster_centers_)

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis', alpha=0.6, label='Clustered Data')
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=200, marker='X', label='Centroids')
plt.title("K-means Clustering of Wisconsin Breast Cancer Data")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()

cm = confusion_matrix(y, y_kmeans)
print("Confusion Matrix:\n", cm)

if cm[0, 0] + cm[1, 1] < cm[0, 1] + cm[1, 0]:
    y_kmeans = 1 - y_kmeans

print(f"Accuracy of K-means clustering: {accuracy_score(y, y_kmeans):.2f}")
