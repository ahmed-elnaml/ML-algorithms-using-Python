X=np.random.randint(0,20,(4,5))

class KPCA:
    theta,p=None,None
    def __init__(self,theta=0,p=2):
        KPCA.theta=theta
        KPCA.p=p

    def poly_kernal(self,X):
        X=np.array(X)
        K=np.zeros((X.shape[0],X.shape[0]))
        for i in np.arange(X.shape[0]):
            for j in np.arange(i,X.shape[0]):
                k_=(np.dot((X[i]).T,X[j])+KPCA.theta)**KPCA.p
                K[i,j]=k_
                K[j,i]=k_
        return K
    def K_center(self,K):
        K=np.array(K)
        one_n=(1/K.shape[0])*np.ones_like(K)
        K_centered=K-np.dot(one_n,K)-np.dot(K,one_n)+one_n*K*one_n
        return K_centered
    def fit_transform(self,X,k_features):
        va,vect=np.linalg.eig(self.K_center(self.poly_kernal(X)))
        transformed_X=vect[np.argsort(np.abs(va),axis=0)][:k_features].T
        return transformed_X
kpca=KPCA(4,3)
print(X)
print(kpca.fit_transform(X, 2))








        






