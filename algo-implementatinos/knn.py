import numpy as np
import math
from collections import Counter

# x, y -> n-dimensional vectors
def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def get_neighbors(train, test_row, k):
    distances = list()
    for train_row in train:
        dst = euclidean_distance(test_row, train_row)
        distances.append((train_row, dst))
    distances.sort(key=lambda x: x[1])

    neighbors = list()
    for i in range(k):
        neighbors.append(distances[i][0])

    return neighbors

def predict_classification(train, test_row, k):
    neighbors = get_neighbors(train, test_row, k)
    class_marks = [row[-1] for row in neighbors]
    predicted_mark = max(set(class_marks), key=class_marks.count)
    return predicted_mark

class KNN:
    # initialize hyperparameter
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)
    
    def _predict(self, x):
        # compute distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # get nearest samples, labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority vote, most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]



# Test distance function
# dataset = [[2.7810836, 2.550537003, 0],
#            [1.465489372, 2.362125076, 0],
#            [3.396561688, 4.400293529, 0],
#            [1.38807019, 1.850220317, 0],
#            [3.06407232, 3.005305973, 0],
#            [7.627531214, 2.759262235, 1],
#            [5.332441248, 2.088626775, 1],
#            [6.922596716, 1.77106367, 1],
#            [8.675418651, -0.242068655, 1],
#            [7.673756466, 3.508563011, 1]]

# new_point = [2.1231231, 1.312312]
# predicted_mark = predict_classification(dataset, new_point, 3)
# print(predicted_mark)