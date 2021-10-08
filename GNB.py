import numpy as np
import math


X=np.random.randint(1,100, size=(1000,6))
y=np.random.randint(0,7, size=(1000,1))
x_test=np.random.randint(1,100, size=(1,6))


class GNB():
    def __init__(self):
        pass
    def fit(self,X,y,x_test):
        X=np.array(X)
        y=np.array(y)
        classes=np.unique(y)
        p_array=np.array([])
        y_count=np.array([len(y[y==class_]) for class_ in classes])[:, np.newaxis]
        for class_ in classes:
            args=np.argwhere(y==class_)[:,0]
            mean_ = np.mean(X[args,:], axis=0)
            var_ = np.var(X[args,:], axis=0)
            p_class=(1/(np.sqrt(2*math.pi*(var_))))*np.exp((x_test-mean_)**2/(-2*var_))
            p_array=np.append(p_class,p_array).reshape(-1,X.shape[1])
        concatenated=np.concatenate((y_count,p_array),axis=1)
        res=np.prod(concatenated,axis=1)
        prob_list= res/sum(res)
        return y[np.argmax(res)]


