import numpy as np
class NaiveBayesClassifier:
    def __init__(self,pdf="Gaussian") -> None:
        self.pdf = pdf

    def fit(self,X_train,y_train):
        unique_labels = np.unique(y_train)
        n = len(X_train[0])
        n_data = len(X_train)
        label_transform = dict([(label,idx) for idx,label in enumerate(unique_labels)])
        self.label_initial = dict([(idx,label) for idx,label in enumerate(unique_labels)])
        data_in_class = [[] for _ in range(len(unique_labels))]
        self.means = np.zeros((len(unique_labels),n))
        self.prob_classes = np.zeros((len(unique_labels),1))
        self.covariance_matrices = np.zeros((len(unique_labels),n,n))
        for idx,instance in enumerate(X_train):
            data_in_class[label_transform[y_train[idx]]].append(instance)
        for idx,data in enumerate(data_in_class):
            data = np.array(data)
            self.means[idx] = np.mean(data,axis=0)
            summ = np.zeros((n,n))
            for element in data:
                summ  = summ + np.outer(element,element) - np.outer(self.means[idx],self.means[idx])
            summ = (1/len(data)) * summ
            self.covariance_matrices[idx,:,:] = summ

    def predict(self,X_test):
        labels = []
        for x in X_test:
            max_d = -np.inf
            max_prob_class = -1
            for idx in range(len(self.means)):
                det_C = np.linalg.det(self.covariance_matrices[idx])
                diff = (x-self.means[idx]).reshape((3,1))
                mult_b = np.matmul(np.linalg.inv(self.covariance_matrices[idx]),diff)
                mult = np.matmul(diff.T,mult_b)
                d = - (1/2)*np.log(det_C) - (1/2)*mult[0][0]
                if(d > max_d):
                    max_d = d
                    max_prob_class = idx
            labels.append(self.label_initial[max_prob_class])
        return labels

