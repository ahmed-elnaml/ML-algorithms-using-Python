import numpy as np
X=np.random.randint(0,20,(4,5))
class LDA_:
    def __init__(self):
        pass
    def fit(self,X,Y):
        X=np.array(X)
        Y=np.array(Y)
        classes = np.unique(Y)
        S_W = np.zeros((X.shape[1], X.shape[1]))
        S_B = np.zeros((X.shape[1], X.shape[1]))
        m=np.mean(X,axis=0)
        for class_ in classes:
            index_ = np.where(Y == class_)
            mi=np.mean(X[index_],axis=0)
            for X_class in X[index_]:
                S_W+=np.dot((X_class-mi)[:,np.newaxis],(X_class-mi)[np.newaxis,:])
            S_B+=(X[Y==class_].shape[1])*np.dot((mi-m)[:,np.newaxis],(mi-m)[np.newaxis,:])
            # eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

        return S_W,S_B
ld=LDA_()
# print(ld.fit(X, Y))









        






