import numpy as np
class LogisticRegression:
    def __init__(self, params):
        """
        figure out necessary params to take as input
        :param params:
        """
        self.learning_rate = params.get('learning_rate', 0.01)
        self.max_iter = params.get('max_iter', 1000)
        self.tolerance = params.get('tolerance', 1e-4)
        # todo: implement

    def sigmoid(self, X, w, b):
        """
        :param X:
        :param w:
        :param b:
        :return: 1 / (1 + np.exp(-(np.dot(X, w) + b)))
        """
        return 1 / (1 + np.exp(-(np.dot(X, w) + b)))

    def gradient(self, X, y, y_pred):
        """
        :param X:
        :param y:
        :param y_pred:
        :return: dw and db
        """
        dw = np.dot(X.T, (y-y_pred)) / y.shape[0]
        db = np.sum(y-y_pred) / y.shape[0]
        return dw, db

    def cross_entropy(self, y, y_pred):
        """
        :param y:
        :param y_pred:
        :return: -sum(y * np.log(y_pred) + (1-y) * np.log(1-y_pred))/nExamples
        """
        return -np.sum(y * np.log(y_pred) + (1-y) * np.log(1-y_pred)) / y.shape[0]


    def gradient_descent(self, X, y):
        X = np.array(X)
        y = np.array(y)

        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        loss_prev = 0

        for i in range(self.max_iter):
            y_pred = self.sigmoid(X, self.weights, self.bias)
            loss = self.cross_entropy(y, y_pred)
            dw, db = self.gradient(X, y, y_pred)
            self.weights += self.learning_rate * dw
            self.bias += self.learning_rate * db

            if(np.isclose(loss, loss_prev, rtol=0,atol=self.tolerance)):
                print('Converged at iteration: ', i)
                break

            loss_prev = loss

            # if i % 100 == 0:
            #     print('loss at iteration: ',i,' is : ', loss)

        # print('final loss: ',loss_prev)



    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        # gradient descent
        self.gradient_descent(X, y)

        assert len(X) == len(y)


    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        :param X:
        :return:
        """
        # todo: implement
        X = np.array(X)
        y_pred = self.sigmoid(X, self.weights, self.bias)
        y_pred = np.where(y_pred > 0.5, 1, 0)

        return y_pred
