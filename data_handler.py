import pandas as pd
import random

def load_dataset():
    """
    function for reading data from csv
    and processing to return a 2D feature matrix and a vector of class
    :return:
    """
    file_name = 'data_banknote_authentication.csv'
    df = pd.read_csv(file_name)

    X = df.drop('isoriginal', axis='columns').values
    y = df['isoriginal'].values
    
    return X,y


def split_dataset(X, y, test_size=0.3, shuffle=False, random_state=None):
    """
    function for spliting dataset into train and test
    :param X:
    :param y:
    :param float test_size: the proportion of the dataset to include in the test split
    :param bool shuffle: whether to shuffle the data before splitting
    :return:
    """
    temp = list(zip(X, y))
    if shuffle:
        if random_state is not None:
            random.seed(random_state)
        random.shuffle(temp)
    X, y = zip(*temp)
    X = list(X)
    y = list(y)

    X_train = X[:int(len(X) * (1 - test_size))]
    y_train = y[:int(len(y) * (1 - test_size))]
    X_test = X[int(len(X) * (1 - test_size)):]
    y_test = y[int(len(y) * (1 - test_size)):]

    return X_train, y_train, X_test, y_test


def bagging_sampler(X, y):
    """
    Randomly sample with replacement
    Size of sample will be same as input data
    :param X:
    :param y:
    :return:
    """
    # todo: implement
    X_sample, y_sample = None, None
    assert X_sample.shape == X.shape
    assert y_sample.shape == y.shape
    return X_sample, y_sample
