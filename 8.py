import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score

data = pd.read_csv('breast_cancer_data.csv')
X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy: {accuracy:.2f}")

print("\nDecision Tree Rules:")
print(export_text(model, feature_names=list(X.columns)))

sample = np.array([
    14.6, 21.7, 94.7, 577.6, 0.102, 0.125, 0.078, 0.057, 0.159, 0.66,
    0.02, 0.039, 0.029, 0.046, 0.063, 0.07, 0.113, 0.144, 0.24, 0.104,
    0.111, 0.51, 0.002, 0.0074, 0.002, 0.007, 0.015, 0.014, 0.027, 0.023
]).reshape(1, -1)

pred = model.predict(pd.DataFrame(sample, columns=X.columns))[0]
label = 'Malignant' if pred == 1 else 'Benign'
print(f"\nPredicted class for new sample: {label}")
