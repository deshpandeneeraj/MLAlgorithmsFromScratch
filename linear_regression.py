import numpy as np
import pandas as pd


class Linear_Regression:
    """
    Basic Linear regression model
    @Author : Neeraj Deshpande
    StartDate : 24th July 2020
    For Single Variable the basic formula is used
    For Multiple Variables, the Matrix formula is used
    Future Improvements:
        -Multiple error metrics
    """
    def __init__(self):
        self.slope = None
        self.intercept = None
        self.parameters = None

    def calculate_slope(self,X ,y):
        print('Fitting for Single Variable')
        #Calculating the averages
        assert len(X) == len(y)
        if type(X) == np.ndarray:
            X = (np.subtract(X, X.min(axis=0))) / (X.max(axis=0) - X.min(axis=0))
        else:
            mini, maxi = min(X), max(X)
            X = np.array([i - mini / maxi - mini for i in X])
        average_X = np.sum(X) / len(X)
        average_y = np.sum(y) / len(y)
        #Calculating the slope
        slope_numerator = 0
        slope_denominator = 0
        for row in range(min(X.shape[0], 50)):
            current = (X[row] - average_X) * (y[row] - average_y)
            slope_numerator += current
            slope_numerator = round(slope_numerator, 3)
            slope_denominator += (X[row] - average_X) ** 2
            slope_denominator = round(slope_denominator, 3)
        self.slope = slope_numerator / slope_denominator

        return average_X, average_y

    def calculate_intercept(self, average_X, average_y):
        try:
            assert self.slope is not None
            self.intercept = average_y - (self.slope * average_X)
            return True
        except AssertionError:
            print(f'calculate_intercept ERROR')
    def fit(self, X, y):
        try:
            if type(y) == np.ndarray: y = y.ravel()
            assert X.shape[0] > 1 and X.shape[0] == y.shape[0]
            if len(X.shape) == 2:
                self.calculate_parameters(X, y)
            elif len(X.shape) == 1:
                average_X, average_y = self.calculate_slope(X, y)
                if self.calculate_intercept(average_X, average_y): print(f'Slope = {self.slope}, Intercept = {self.intercept}')
        except AssertionError:
            print(f'FIT ERROR: Error on length or shape of input : {X.shape[0], y.shape[0]}')

    def calculate_parameters(self, X, y):
        try:
            print('Fitting for multiple Variables')
            assert X.shape[0] == y.shape[0]
            ones = np.ones(shape=(X.shape[0],1))
            X = np.matrix(np.concatenate((ones, X), axis=1), dtype=np.int32)
            X_transpose = X.transpose()
            dot_product = X_transpose.dot(X)
            denominator = np.linalg.inv(dot_product)
            first_dot = np.dot(denominator, X_transpose)
            b = np.dot(first_dot, y)
            self.intercept = b[0]
            self.parameters = b.reshape((b.shape[-1],1))
            print(f'Parameter vector = {self.paramters}')
        except AssertionError:
            print(f"calculate_parameters ERROR")

    def predict_one(self, X):
        try:
            if type(X) != int and type(X) != np.ndarray: raise TypeError(type(X),X)
            if type(X) == np.ndarray:
                if X.shape[0] > 1:
                    X = np.hstack((1, X))
                    X = np.reshape(X, (1, X.shape[0]))
                    prediction = int(np.dot(X, self.parameters))
                elif X.shape[0] == 1:
                    prediction = (self.slope * X) + self.intercept
                else: raise AssertionError()
            else:
                prediction = (self.slope * X) + self.intercept
            return prediction
        except AssertionError:
            print('predict_one AssertionERROR', X.shape)

    def predict_many(self, X):
        # if type(X) == np.ndarray: X = X.ravel()
        predictions = []
        for value in X:
            predictions.append(self.predict_one(value))
        return predictions


if __name__ == "__main__":
    from sklearn.metrics import explained_variance_score
    #Single Variable
    train = pd.read_csv('data/linear_train.csv')
    test = pd.read_csv('data/linear_test.csv')
    X_train, y_train, X_test, y_test = train.x, train.y, test.x, test.y
    #Multiple Variables
    X = np.array([[110,40],[120,30], [100, 20], [90, 0],[80, 10]])
    y = np.array([100, 90, 80, 70, 60])
    linreg = Linear_Regression()
    linreg.fit(X_train, y_train)
    predictions = linreg.predict_many(X_test)
    for i, j in list(zip(y_test, predictions))[:5]:
        print(f'{i} VS {j} = {i-j}')
    accuracy = explained_variance_score(y_test, predictions)
    print(f'Accuracy = {round(accuracy*100, 2)}%')
