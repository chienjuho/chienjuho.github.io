#!/usr/bin/python3
# Homework 3 Code
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler

def find_test_error(w, X, y):
    # find_test_error: compute the test error of a linear classifier w.
    # The hypothesis is assumed to be of the form sign([1 x(n,:)] * w)
    # Inputs:
    #        w: weight vector
    #        X: data matrix (without an initial column of 1s)
    #        y: data labels (plus or minus 1)
    # Outputs:
    #        test_error: binary classification error of w on the data set (X, y)
    #        this should be between 0 and 1.

    # Your code here, assign the proper value to test_error:

    # Initialize variables
    N = X.shape[0]

    # Add column of ones to X
    X_adj = np.hstack([np.ones(N).reshape(-1, 1), X])

    # Get predictions
    preds = np.sign(np.matmul(X_adj, w))

    # Reshape y
    y = y.reshape(-1, 1)

    # Find error
    test_error = np.sum(preds != y) / N

    return test_error


def logistic_reg(X, y, w_init, max_its, eta):
    # logistic_reg learn logistic regression model using gradient descent
    # Inputs:
    #        X : data matrix (without an initial column of 1s)
    #        y : data labels (plus or minus 1)
    #        w_init: initial value of the w vector (d+1 dimensional)
    #        max_its: maximum number of iterations to run for
    #        eta: learning rate
    # Outputs:
    #        t : number of iterations gradient descent ran for
    #        w : weight vector
    #        e_in : in-sample error (the cross-entropy error as defined in LFD)

    # Your code here, assign the proper values to t, w, and e_in:

    # Initialize variables
    N = X.shape[0]
    threshold = 10 ** (-6)

    # Add column of ones to X
    X_adj = np.hstack([np.ones(N).reshape(-1, 1), X])

    # Reshape to single column matrix
    w = w_init.reshape(-1, 1)
    y = y.reshape(-1, 1)

    # Run gradient descent
    if max_its == None:
        # For part B
        t = 0
        v = np.repeat(np.inf, N)
        threshold = 10 ** (-6)

        while (any(abs(v) > threshold)):
            v = np.sum((-y * X_adj) / (1 + np.exp(y * np.matmul(X_adj, w))), axis=0).T / N
            v = v.reshape(-1, 1)
            w = w - eta * v
            t = t + 1
    else:
        # For part A
        for t in range(0, max_its + 1):
            v = np.sum((-y * X_adj) / (1 + np.exp(y * np.matmul(X_adj, w))), axis=0).T / N
            v = v.reshape(-1, 1)
            w = w - eta * v
            if all(abs(v) < threshold):
                break

    # Calculate loss
    e_in = np.sum(np.log(1 + np.exp(-y * np.matmul(X_adj, w))), axis=0) / N

    return t, w, e_in


def logistic_reg_regularizer(X, y, w_init, max_its, eta, reg, penalty):
    # logistic_reg learn logistic regression model using gradient descent with L1 or L2 regularizer
    # Inputs:
    #        X : data matrix (without an initial column of 1s)
    #        y : data labels (plus or minus 1)
    #        w_init: initial value of the w vector (d+1 dimensional)
    #        max_its: maximum number of iterations to run for
    #        eta: learning rate
    #        reg: lambda for regularizer
    #        penalty: specify the norm used in the penalization, L1 or L2
    # Outputs:
    #        t : number of iterations gradient descent ran for
    #        w : weight vector
    #        e_in : in-sample error (the cross-entropy error as defined in LFD)

    # Your code here, assign the proper values to t, w, and e_in:

    # Initialize
    N = X.shape[0]
    threshold = 10 ** (-6)
    X_adj = np.hstack([np.ones(N).reshape(-1, 1), X])
    w = w_init.reshape(-1, 1)
    y = y.reshape(-1, 1)

    # Run gradient descent
    if penalty == "L1":
        for t in range(0, max_its + 1):
            v = np.sum((-y * X_adj) / (1 + np.exp(y * np.matmul(X_adj, w))), axis=0).T / N
            v = v.reshape(-1, 1)
            w = w - eta * v
            w_new = w - eta * reg * np.sign(w)
            w_new[np.sign(w_new) != np.sign(w)] = 0
            w = w_new
            if all(abs(v) < threshold):
                break
    elif penalty == "L2":
        for t in range(0, max_its + 1):
            v = np.sum((-y * X_adj) / (1 + np.exp(y * np.matmul(X_adj, w))), axis=0).T / N
            v = v.reshape(-1, 1) + 2 * reg * w
            w = w - eta * v
            if all(abs(v) < threshold):
                break

    e_in = np.sum(np.log(1 + np.exp(-y * np.matmul(X_adj, w))), axis=0) / N

    return t, w, e_in



def main_cleveland():
    # Load training data
    train_data = pd.read_csv('clevelandtrain.csv')

    # Parse training data
    X_train = np.array(train_data.iloc[:, :-1])
    y_train = np.array(train_data.iloc[:, -1])
    y_train[y_train == 0] = -1

    # Load test data
    test_data = pd.read_csv('clevelandtest.csv')

    # Parse test data
    X_test = np.array(test_data.iloc[:, :-1])
    y_test = np.array(test_data.iloc[:, -1])
    y_test[y_test == 0] = -1

    # Initialize weight vector
    N = X_train.shape[1]
    w_init = np.zeros(N + 1)

    std_scale = StandardScaler().fit(X_train)
    X_train_zscore = std_scale.transform(X_train)
    X_test_zscore = std_scale.transform(X_test)
    learning_rates = 0.01
    reg_lambdas = [0, 0.001, 0.01, 0.05, 0.1]
    max_iterations = 10 ** 6

    # baseline
    eta = learning_rates
    tic = time.perf_counter()
    t, w, e_in = logistic_reg(X_train_zscore, y_train, w_init, max_iterations, eta)
    train_time = time.perf_counter() - tic
    test_error = find_test_error(w, X_test_zscore, y_test)
    print(eta, t, e_in[0], test_error, train_time)

    # L2 loss
    for reg in reg_lambdas:
        t, w, e_in = logistic_reg_regularizer(X_train_zscore, y_train, w_init, max_iterations, eta, reg, "L2")
        test_error = find_test_error(w, X_test_zscore, y_test)
        train_time = time.perf_counter() - tic
        print("L2 reg ", eta, reg, t, e_in[0], test_error, str(train_time) + " seconds",
              "0vec=" + str(np.sum(np.abs(w) <= 10**(-5))))

    # L1 loss
    for reg in reg_lambdas:
        t, w, e_in = logistic_reg_regularizer(X_train_zscore, y_train, w_init, max_iterations, eta, reg, "L1")
        test_error = find_test_error(w, X_test_zscore, y_test)
        train_time = time.perf_counter() - tic
        # print(w)
        print("L1 reg ", eta, reg, t, e_in[0], test_error, str(train_time) + " seconds",
              "0vec=" + str(np.sum(np.abs(w) <= 10**(-5))))



def main_digits():
    # Load data
    X_train, X_test, y_train, y_test = np.load("digits_preprocess.npy", allow_pickle=True)
    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1

    N = X_train.shape[1]
    w_init = np.zeros(N + 1)

    std_scale = StandardScaler().fit(X_train)
    X_train_zscore = std_scale.transform(X_train)
    X_test_zscore = std_scale.transform(X_test)
    learning_rates = 0.01
    reg_lambdas = [ 0.0001, 0.001,0.005, 0.01, 0.05, 0.1]
    max_iterations = 10 ** 4

    eta = learning_rates
    tic = time.perf_counter()
    t, w, e_in = logistic_reg(X_train_zscore, y_train, w_init, max_iterations, eta)
    train_time = time.perf_counter()- tic
    test_error = find_test_error(w, X_test_zscore, y_test)
    print("baseline ", eta, t, e_in[0], test_error, str(train_time) + "seconds",
          "0vec=" + str(np.sum(np.abs(w) <= 10**(-6))))

    # L2 loss
    for reg in reg_lambdas:
        t, w, e_in = logistic_reg_regularizer(X_train_zscore, y_train, w_init, max_iterations, eta, reg, "L2")
        test_error = find_test_error(w, X_test_zscore, y_test)
        train_time = time.perf_counter() - tic
        print("L2 reg ", eta, reg, t, e_in[0], test_error, str(train_time) + "seconds",
              "0vec="+str(np.sum(np.abs(w) <= 10**(-6))))

    # L1 loss
    for reg in reg_lambdas:
        t, w, e_in = logistic_reg_regularizer(X_train_zscore, y_train, w_init, max_iterations, eta, reg, "L1")
        test_error = find_test_error(w, X_test_zscore, y_test)
        train_time = time.perf_counter() - tic
        print("L1 reg ", eta, reg, t, e_in[0], test_error, str(train_time) + "seconds",
              "0vec="+str(np.sum(np.abs(w) <= 10**(-6))))


if __name__ == "__main__":
    #main_cleveland()
    main_digits()
