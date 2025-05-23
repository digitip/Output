import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

np.random.seed(42)
x = np.linspace(0, 1, 100)[:, None]
y = 2 * x.ravel()**2 + 0.3 * np.random.randn(100)

def lwr(x_train, y_train, tau, x_test):
    preds = []
    for x in x_test:
        w = np.exp(-np.sum((x_train - x)**2, axis=1) / (2 * tau**2))
        W = np.diag(w)
        theta = inv(x_train.T @ W @ x_train) @ (x_train.T @ W @ y_train)
        preds.append(x @ theta)
    return np.array(preds)

x_test = np.linspace(0, 1, 200)[:, None]
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red', label='Data points')

for tau in [0.1, 0.3, 0.5]:
    y_pred = lwr(x, y, tau, x_test)
    plt.plot(x_test, y_pred, label=f'Tau={tau}')

plt.title("Locally Weighted Regression (LWR)")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
