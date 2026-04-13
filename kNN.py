import numpy as np
from collections import Counter

class Knn:
    def __init__(self, k=5):
        self.n_neighbors = k

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def predict(self, X_test):
        X_test = np.array(X_test)
        y_pred = []

        for i in X_test:
            distances = []
            for j in self.X_train:
                distances.append(self.calculate_distance(i, j))

            n_neighbors = sorted(
                list(enumerate(distances)),
                key=lambda x: x[1]
            )[:self.n_neighbors]

            label = self.majority_count(n_neighbors)
            y_pred.append(label)

        return np.array(y_pred)

    def calculate_distance(self, point_A, point_B):
        return np.linalg.norm(point_A - point_B)

    def majority_count(self, n_neighbors):
        votes = [self.y_train[i[0]] for i in n_neighbors]
        return Counter(votes).most_common(1)[0][0]