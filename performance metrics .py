import numpy as np
class Predictor:
    def __init__(self):
        pass
    def accuracy_score(self,Y,Y_predicted):
            Y=np.array(Y)
            Y_predicted=np.array(Y_predicted)
            if Y.shape[0]==Y_predicted.shape[0]:
                out= sum(Y==Y_predicted)/(Y.shape[0])
                return round(out,4)
            else:
                raise Exception(" Dimension Incomptability")
    def balanced_score(self,Y,Y_predicted,labels=None):
        Y = np.array(Y)
        Y_predicted = np.array(Y_predicted)
        if Y.shape[0] == Y_predicted.shape[0]:
            classes=np.unique(Y)
            out=np.array([])
            for class_ in classes :
                index_=np.where(Y==class_)
                class_score=sum(Y[index_] == Y_predicted[index_]) / (Y[index_].shape[0])
                out=np.append(out,class_score)
            if labels==None:
                return round(out.mean(),4),out.astype(np.float_)
            else:
                return round(out.mean(),4), dict(zip(labels,out))
        else:
            raise Exception(" Dimension Incomptability")


    def conf_mat(self,Y,Y_predicted,labels=None):
        Y = np.array(Y)
        Y_predicted = np.array(Y_predicted)
        if Y.shape[0] == Y_predicted.shape[0]:
            classes = np.unique(Y)  
            out = np.array([])
            for class_ in classes:
                index_ = np.where(Y == class_)
                row=np.array([])
                for pred_class in classes:
                    row=np.append(row,sum(Y_predicted[index_]==pred_class))
                print(row)
                out=np.append(out,row)
            return out.reshape(classes.shape[0],classes.shape[0]).astype(int)




Y=np.random.randint(0,4,20)
Y_predicted=np.random.randint(0,6,100)
pr=Predictor()
# print(pr.balanced_score(Y, Y_predicted))
# print(pr.conf_mat(Y, Y_predicted))









        






