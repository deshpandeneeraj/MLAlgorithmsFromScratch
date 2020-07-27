import numpy as np
import pandas as pd
import math



class K_Nearest_Neighbours:
    """
        ***Simple Implementation of KNN Classification algorithm***
        @Author : Neeraj Deshpande
        StartDate : 24 July 2020
        Default distance used is Euclidean distance
        Functionality provided for normalization
        Future Improvements:
            - Variable P-Value in Minkowski distance measure
    """
    def __init__(self, k):
        self.k = k
        print(f'K Nearest Neighbour Model initialized with K = {self.k}')
        if k % 2 == 0:
            print(f'***WARNING: K value is even, result accuracy may suffer for Binary Classification***')
        self.X_values:np.ndarray = None
        self.y_labels:np.ndarray = None
        self.y_map = None

    @staticmethod
    def y_normalize(y_labs:np.ndarray):
        try:
            #Check input shape/length
            assert len(y_labs) > 1
            #Attempt to fix illegal shape
            if len(y_labs.shape) > 1 :
                y_labs = y_labs.ravel()
                print(f'Shape of input possibly incorrect, input.ravel() applied')
            y_out = np.zeros(shape=(y_labs.shape),dtype=np.uint8)
            labels = list(set(y_labs))
            #Provide Mapping for y_label output
            label_map = {labels.index(i):i for i in labels}
            for index in range(y_labs.shape[0]):
                y_out[index] = labels.index(y_labs[index])

            return y_out, label_map

        except AssertionError as e:
            print(f'y_normalize ERROR: Input error, Check shape or length of input : {y_labs.shape}')

    @staticmethod
    def X_normalize(X_vals:np.ndarray):
        try:
            X = pd.DataFrame(X_vals)
            #Check if all columns are of numeric datatypes
            numeric_dtypes = [np.dtype('float64'), np.dtype('int32')]
            assert X.apply(lambda c: c.dtype).isin(numeric_dtypes).all()
            #Simple MinMax normalization for now
            X = (X - X.min()) / (X.max() - X.min())

            return X.values

        except AssertionError as e:
            print(f'X_normalize ERROR: Possibly not all collumns are numeric :\n{X.dtypes}')

    def fit(self, X, y):
        try:
            #Checking count of datapoints
            assert X.shape[0] == y.ravel().shape[-1]
            #Normalizing X and y, if X is already normalized - it will not be affected by MinMax normalization
            self.y_labels, self.y_map = self.__class__.y_normalize(y)
            # self.X_values = self.__class__.X_normalize(X)
            self.X_values = X
            print(f'Fitted on {len(self.y_labels)} datapoints')
        except AssertionError:
            print(f'Fitting ERROR: Lengths of X and y are different : X={X.shape[-1]}, y={y.ravel().shape[-1]}')


    @staticmethod
    def calculate_distance(X_input, X_fitted):
        try:
            difference = 0
            X_input = X_input.ravel()
            #Check if input shape matches that of training data shape
            assert X_input.shape[-1] == X_fitted.shape[-1]
            for col in range(X_fitted.shape[-1]):
                #Euclidean distance
                difference += (X_input[col] - X_fitted[col]) ** 2
            distance = np.sqrt(difference)
            return distance
        except AssertionError:
            print(f"calculate_distance ERROR: {X_input.shape} vs {X_fitted.shape}")

    @staticmethod
    def get_neighbour(distance_matrix, k):
        distance_matrix.sort()
        #Select closest K datapoints
        closest_labels = [i[-1] for i in distance_matrix[:k]]
        counts = [[i, closest_labels.count(i)] for i in set(closest_labels)]
        counts.sort(reverse=True)
        #Return highest count noraalized label
        return counts[0][0]

    def calculate_distance_matrix(self, X_input:np.ndarray):
        try:
            X_input = X_input.ravel()
            assert X_input.shape[-1] == self.X_values.shape[-1]
            assert type(X_input) == np.ndarray
            distance_matrix = []
            #Prepare the distance matrix
            for index in range(len(self.X_values)):
                distance_matrix.append([self.__class__.calculate_distance(X_input, self.X_values[index]), self.y_labels[index]])
            return distance_matrix
        except AssertionError:
            print(f'calculate_distance_matrix ERROR: Error in input shape or Numpy array not given: Given shape = {X_input.shape} , Expected = {self.X_values.shape}, Type = {type(X_input)}')

    def predict_one(self, X_input:np.ndarray):
        #Check for right number of input variables
        distance_matrix = self.calculate_distance_matrix(X_input)
        neighbour = self.__class__.get_neighbour(distance_matrix, self.k)
        return self.y_map.get(neighbour)

    def predict_many(self, X_inputs:np.ndarray):
        predictions = []
        for index in range(X_inputs.shape[0]):
            predictions.append(self.predict_one(X_inputs[index]))
        return predictions


if __name__ == "__main__":
    #Testing imports
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    import timeit
    data_raw = pd.read_csv('data/Iris.csv')
    X = data_raw.iloc[:, 1:-1].values
    # X = K_Nearest_Neighbours.X_normalize(X)
    y = data_raw.iloc[:,-1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    start = timeit.default_timer()

    knn = K_Nearest_Neighbours(4)
    knn.fit(X = X_train,y=y_train)
    predictions = knn.predict_many(X_test)

    stop = timeit.default_timer()
    print(f'\nNeeraj Accuracy = {accuracy_score(y_test, predictions)}')

    print(f'\nNeeraj Runtime: {(stop - start)*1000} milliseconds')

    start = timeit.default_timer()

    knn = KNeighborsClassifier(n_neighbors=4)
    knn.fit(X_train,y_train)
    predictions = knn.predict(X_test)
    print(f'\nSKLearn Accuracy = {accuracy_score(y_test, predictions)}')

    stop = timeit.default_timer()
    print(f'\nSKLearn Runtime: {(stop - start)*1000} milliseconds')
