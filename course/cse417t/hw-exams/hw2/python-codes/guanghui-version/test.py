#!/usr/bin/python3
# Homework 2 Code
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


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
    threshold = 10 ** (-3)

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


def glmfit_example(X_train, y_train, X_test, y_test):
    # logistic regression from sklearn LogisticRegression
    N = X_train.shape[0]
    clf = LogisticRegression(max_iter=1e6, penalty='none').fit(X_train, y_train)
    w = clf.coef_.reshape(-1, 1)
    e_in = np.sum(np.log(1 + np.exp(-y_train.reshape(-1, 1) * (np.matmul(X_train, w) + clf.intercept_)))) / N
    train_error = 1 - clf.score(X_train, y_train)
    test_error = 1 - clf.score(X_test, y_test)

    # version of find_test_error function
    # w = np.vstack((clf.intercept_, w))
    # train_error = find_test_error(w, X_train, y_train)
    # test_error = find_test_error(w, X_test, y_test)
    
    print("logistic regression", e_in, train_error, test_error)
    return w, e_in


def main():
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

    # Set parameters
    learning_rate = 10 ** (-5)
    max_iterations = [10 ** 4, 10 ** 5, 10 ** 6]

    # Initialize weight vector
    N = X_train.shape[1]
    w_init = np.zeros(N + 1)

    glmfit_example(X_train, y_train, X_test, y_test)
    return;

    # Part A
    print('Part A:')
    # Train
    for iterations in max_iterations:
        # Run logistic regression and time it
        tic = time.perf_counter()
        t, w, e_in = logistic_reg(X_train, y_train, w_init, iterations, learning_rate)
        toc = time.perf_counter()
        train_time = toc - tic

        # Calculate errors
        train_error = find_test_error(w, X_train, y_train)
        test_error = find_test_error(w, X_test, y_test)

        print(iterations, e_in[0], train_error, test_error, train_time)

    glmfit_example(X_train, y_train, X_test, y_test)

    # Part B
    print('Part B:')
    std_scale = StandardScaler().fit(X_train)
    X_train_zscore = std_scale.transform(X_train)
    X_test_zscore = std_scale.transform(X_test)
    learning_rates = [0.01, 0.1, 1, 4, 6, 7]

    # Train
    for eta in learning_rates:
        # Run logistic regression and time it
        tic = time.perf_counter()
        t, w, e_in = logistic_reg(X_train_zscore, y_train, w_init, None, eta)
        toc = time.perf_counter()
        train_time = toc - tic

        # Calculate errors
        test_error = find_test_error(w, X_test_zscore, y_test)

        print(eta, t, e_in[0], test_error, train_time)


if __name__ == "__main__":
    main()
