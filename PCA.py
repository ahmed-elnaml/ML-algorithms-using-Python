import numpy as np

   



def corr_mat(X):
    X=np.array(X)
    sig_sig=np.array([])
    for i in np.arange(X.shape[1]):
        for j in np.arange(X.shape[1]):
            sig_sig=np.append(sig_sig,np.std(X[:,i])*np.std(X[:,j]))
    sig_sig=sig_sig.reshape(X.shape[1],-1)
    return (cov_mat(X)/sig_sig).astype(dtype=np.float16)

X=np.random.randint(0,20,(4,5))
# X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
# print(cov_mat(X))
print(X)
class Pca:
    def __init__(self,X):
        self.X=X

    def cov_mat(X):
        X = np.array(X)
        mat = np.array([])
        for j in np.arange(X.shape[1]):
            for i in np.arange(X.shape[1]):
                var_ = np.dot((X[:, j] - X[:, j].mean()), (X[:, i] - X[:, i].mean())) / (X.shape[0])
                mat = np.append(mat, var_)
        return mat.reshape(X.shape[1], -1)



    def fit_transform(self,k):
        # print(X)
        # print(self.X)
        va,vect=np.linalg.eig(cov_mat(self.X))
        w_entire=vect[np.argsort(np.abs(va),axis=0)]
        w=(np.array(w_entire)[:k]).T
        return (np.dot(self.X,w)).astype(dtype=np.float_)
compressor=Pca(X)
# print(compressor.fit_transform(6))
