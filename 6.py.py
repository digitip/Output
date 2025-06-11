import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
np.random.seed(42)
x=np.linspace(0,1,100)
y=2*x**2+0.3*np.random.randn(100)
x=x[:,np.newaxis]

def locally_weighted_regression(x_train,y_train,tau,x_test):
    m,n= x_train.shape
    y_pred=np.zeros(len(x_test))
    for i,x in enumerate(x_test):
        weights=np.exp(-np.sum((x_train-x)**2,axis=1)/(2*tau**2))
        W=np.diag(weights)
        X_TW_X=x_train.T@W@x_train
        X_TW_Y=x_train.T@W@y_train
        theta=inv(X_TW_X)@X_TW_Y
        y_pred[i]=x@theta
    return y_pred
tau_values=[0.1,0.3,0.5]
x_test=np.linspace(0,1,200)[:,np.newaxis]
plt.figure(figsize=(10,6))
plt.scatter(x, y, color="red", label='Data points')

for tau in tau_values:
    y_pred=locally_weighted_regression(x, y, tau, x_test)
    plt.plot(x_test,y_pred,label=f'Tau={tau}')
plt.title("Locally weighted Regression (LWR)")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
   