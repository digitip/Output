import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('California Housing.csv')

print("First row of the dataset:")
print(df.head(1))
print(f"Dataset shape: {df.shape}")

df.hist(bins=20, figsize=(15, 10), color='skyblue', edgecolor='black')
plt.suptitle("Histograms for Numerical Features", fontsize=16)
plt.show()

numeric_cols = df.select_dtypes(include='number').columns
for col in numeric_cols:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df[col], color='lightgreen')
    plt.title(f"Boxplot of {col}", fontsize=14)
    plt.show()

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"Feature: {col}")
    print(f"Number of outliers: {len(outliers)}")
    print(f"Outlier values:\n{outliers[col].values}\n")
