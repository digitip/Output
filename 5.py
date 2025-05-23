import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

values = np.random.rand(100)
labels = ['Class1' if v <= 0.5 else 'Class2' for v in values[:50]] + [None] * 50

df = pd.DataFrame({
    "Point": [f"x{i+1}" for i in range(100)],
    "Value": values,
    "label": labels
})

df["Value"].hist(bins=10, edgecolor='black', figsize=(12, 8))
plt.title("Frequency for Value", fontsize=16)
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

train_df = df.dropna()
x_train, y_train = train_df[["Value"]], train_df["label"]
x_test = df[df["label"].isna()][["Value"]]
true_labels = ['Class1' if v <= 0.5 else 'Class2' for v in values[50:]]

k_list = [1, 2, 3, 4, 5, 20, 30]
for k in k_list:
    model = KNeighborsClassifier(n_neighbors=k).fit(x_train, y_train)
    preds = model.predict(x_test)
    acc = accuracy_score(true_labels, preds) * 100
    print(f"Accuracy for k={k}: {acc:.2f}%")
    df.loc[df["label"].isna(), f"label_k{k}"] = preds

result_df = df[df["label"].isna()].drop(columns="label")
print("\nDataFrame with 'label':")
print(result_df)
