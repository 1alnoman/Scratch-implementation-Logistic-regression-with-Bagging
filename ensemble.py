from data_handler import bagging_sampler
import random
import copy
import numpy as np


class BaggingClassifier:
    def __init__(self, base_estimator, n_estimator):
        """
        :param base_estimator:
        :param n_estimator:
        :return:
        """
        self.base_estimator = base_estimator
        self.n_estimator = n_estimator
        self.bootstrap_Xs = []
        self.bootstrap_ys = []
        self.models = []
        # todo: implement

    def fit(self, X, y,random_state=None):
        """
        :param X:
        :param y:
        :return: self
        """

        # fit all n_estimator models
        if random_state is not None:
            random.seed(random_state)
        self.bootstrap_sampling(X, y)
        self.create_models()
        for i in range(self.n_estimator):
            self.models[i].fit(self.bootstrap_Xs[i], self.bootstrap_ys[i])
        # assert X.shape[0] == y.shape[0]
        # assert len(X.shape) == 2
        # todo: implement

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        apply majority voting
        :param X:
        :return:
        """
        y_all = []
        for i in range(self.n_estimator):
            y_all.append(self.models[i].predict(X))

        y_all = np.array(y_all)

        y_pred = []
        for i in range(len(X)):
            y_pred.append(np.bincount(y_all[:, i]).argmax())

        return y_pred
        # todo: implement

    # sample with repeat n_estimator times from X and y
    def bootstrap_sampling(self, X, y):
        """
        :param X:
        :param y:
        :return:
        """
        len_data = len(X)
        data_index = [random.choices(range(len_data),k=len_data) for _ in range(self.n_estimator)]

        for i in range(self.n_estimator):
            self.bootstrap_Xs.append([X[i] for i in data_index[i]])
            self.bootstrap_ys.append([y[i] for i in data_index[i]])

    # create n_estimator models
    def create_models(self):
        """
        :return:
        """
        for i in range(self.n_estimator):
            self.models.append(copy.deepcopy(self.base_estimator))
    

