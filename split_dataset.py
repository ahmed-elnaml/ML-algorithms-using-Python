import numpy as np
import math
def split_dataset(X,y,test_size=0.2,seeds=None,stratify=None):
    if X.shape[0]!=y.shape[0]:
        print("check your dimensions")
    np.random.seed(seeds)
    index_=np.random.permutation(X.shape[0])
    train_size=int(X.shape[0]*(1-test_size))
    print("train_size", train_size)
    X_train,y_train=X[index_[:train_size]],y[index_[:train_size]]
    X_test,y_test=X[index_[train_size:]],y[index_[train_size:]]
    return X_train, X_test, y_train.T, y_test.T

X=np.random.randint(1,100, size=(1000,6))
y=np.random.randint(0,7, size=(1000,1))
x_test=np.random.randint(1,100, size=(1,6))


X_train, X_test, y_train, y_test=split_dataset(X, y, test_size=0.25,seeds=5)
# print(X_train)
# print(y_train)






