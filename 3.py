import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

df = pd.read_csv("iris_data.csv")
features = df.columns[:4]

scaled = StandardScaler().fit_transform(df[features])

cov = np.cov(scaled.T)
print("Cov Matrix:\n", cov)

vals, vecs = np.linalg.eig(cov)
print("Eigenvalues:", vals)
print("Eigenvectors:\n", vecs)

pca = PCA(n_components=2)
components = pca.fit_transform(scaled)

df_pca = pd.DataFrame(components, columns=['PC1', 'PC2'])
df_pca['Species'] = df['Species']

sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Species', palette='Set1', s=100)
plt.title('PCA of IRIS Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()
