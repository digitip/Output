import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = pd.read_csv('olivetti_faces.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

pca = PCA(n_components=50)
X_pca = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy of NaiveBayes classifier: {accuracy_score(y_test, y_pred):.2f}")

for i in [0, 5, 10]:
    pred = model.predict([X_test[i]])[0]
    print(f"\nPredicted label for {i}: {pred}")
    img = X[i].reshape(64, 64)
    plt.imshow(img, cmap='gray')
    plt.title(f"True label: {y_test[i]}, Predicted label: {pred}")
    plt.show()
