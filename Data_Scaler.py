import numpy as np
import math

X=np.random.randint(1,100, size=(1000,6))
y=np.random.randint(0,7, size=(1000,1))
x_test=np.random.randint(1,100, size=(1,6))

class Data_Scaler:
    def __init__(self,method="min_max"):
        self.method=method
    def fit(self,X):
        X=np.array(X)
        if self.method=="min_max":
            return (X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))
        elif self.method=="stand":
            return (X-X.mean(axis=0))/X.std(axis=0)
# ds=Data_Scaler(method="stand")
# print(ds.fit(X))


