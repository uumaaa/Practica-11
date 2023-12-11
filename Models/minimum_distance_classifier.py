import numpy as np
class MinimumDistanceClassifier:
    def __init__(self,classes) -> None:
        self.classes = classes

    def fit(self,X_train,Y_train):
        unique_labels = np.unique(Y_train)
        label_transform = dict([(label,idx) for idx,label in enumerate(unique_labels)])
        self.label_initial = dict([(idx,label) for idx,label in enumerate(unique_labels)])
        data_in_class = [[] for _ in range(self.classes)]
        self.centroids = [[] for _ in range(self.classes)]
        for idx,instance in enumerate(X_train):
            data_in_class[label_transform[Y_train[idx]]].append(instance)
        for idx,data in enumerate(data_in_class):
            self.centroids[idx] = np.mean(data,0)
    
    def predict(self,X_test):
        distances = np.linalg.norm(X_test[:, np.newaxis, :] - self.centroids, axis=2)
        pred = np.argmin(distances, axis=1)
        predictions = [self.label_initial[pred[i]] for i in range(len(pred))]
        return predictions