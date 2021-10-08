import numpy as np
import math

X=np.random.randint(1,100, size=(1000,6))
y=np.random.randint(0,7, size=(1000,1))
x_test=np.random.randint(1,100, size=(1,6))


class KNN_Classifer():
    def __init__(self,n=1):
        self.n=n
    def fit(self,X,y,x_test):
        return y[np.argsort(np.array([np.linalg.norm(x - x_test) for x in X]))[:-self.n-1:-1]].reshape(-1)
    def predict(self,X,y,x_test):
        return np.bincount(self.fit(X,y,x_test)).argmax()

class KNN_regressor(KNN_Classifer):
    def predict(self,X,y,x_test):
        return np.mean(self.fit(X,y,x_test))
# knn_r=KNN_regressor(5)
# print(knn_r.predict(X, y, x_test))








