import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load data
df = pd.read_csv("iris_data.csv")

# Features to use
features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# Standardize the features (simplified)
X = StandardScaler().fit_transform(df[features])

# Covariance matrix and eigen values/vectors (simplified)
print("\nCovariance Matrix:\n", np.cov(X.T))
vals, vecs = np.linalg.eig(np.cov(X.T))
print("\nEigenvalues:", vals)
print("\nEigenvectors:\n", vecs)

# Apply PCA
pca = PCA(n_components=2)
pcs = pca.fit_transform(X)

# Create DataFrame with PCA results
df_pca = pd.DataFrame(pcs, columns=['PC1', 'PC2'])
df_pca['Species'] = df['Species']

# Plot PCA result (simplified)
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Species')
plt.title("PCA of Iris Dataset")
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title="Species")
plt.show()
