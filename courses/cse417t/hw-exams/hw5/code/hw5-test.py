#!/usr/bin/python3
# Homework 5 Code
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt

def adaboost_trees(X_train, y_train, X_test, y_test, n_trees, title = "Nan"):
    # %AdaBoost: Implement AdaBoost using decision trees
    # %   using decision stumps as the weak learners.
    # %   X_train: Training set
    # %   y_train: Training set labels
    # %   X_test: Testing set
    # %   y_test: Testing set labels
    # %   n_trees: The number of trees to use

    N = X_train.shape[0]
    d = np.ones(N) / N

    alpha_ts = np.zeros(n_trees)
    train_predictions = np.zeros([N, n_trees])
    test_predictions = np.zeros([X_test.shape[0], n_trees])
    train_error = np.zeros(n_trees)
    test_error = np.zeros(n_trees)

    for t in range (n_trees):
        decision_tree = DecisionTreeClassifier(criterion='entropy', max_depth=1)
        decision_tree.fit(X_train, y_train, sample_weight=d)

        train_predictions[:, t] = decision_tree.predict(X_train)
        test_predictions[:, t] = decision_tree.predict(X_test)
        e_ts = np.sum(d * (train_predictions[:, t] != y_train))
        print(train_error[t])
        alpha_ts[t] = np.log( (1 - e_ts) /e_ts ) / 2
        d = d * np.exp( -alpha_ts[t] * y_train * train_predictions[:, t] )
        d = d / np.sum(d)
        print(t, str(e_ts),alpha_ts[t])

        ada_train_pred = np.sign( np.matmul( train_predictions[:,:t+1] ,alpha_ts[:t+1]) )
        train_error[t] = 1.0 * np.sum(ada_train_pred != y_train) / X_train.shape[0]
        ada_test_pred = np.sign( np.matmul( test_predictions[:,:t+1] , alpha_ts[:t+1]) )
        test_error[t] = 1.0 * np.sum(ada_test_pred != y_test) / X_test.shape[0]
        print(t, train_error[t],test_error[t], ada_train_pred[1],ada_train_pred[2])

    line_train,  = plt.plot(np.arange(1, n_trees + 1), train_error, label='train error')
    line_test,  = plt.plot(np.arange(1, n_trees + 1), test_error, label='test error')
    plt.legend(handles=[line_train, line_test])
    plt.xlabel('Number of trees')
    plt.ylabel('prediction error')
    plt.title('adaboost ' + title)
    plt.show()
    return train_error, test_error


def subdata(label1, label2, og_train, og_test, title = 'nan'):
    # Find labels
    one_vs_three = np.where(og_train[:, 0] == label1)[0].tolist() + np.where(og_train[:, 0] == label2)[0].tolist()
    train_data = og_train[one_vs_three, :]
    one_vs_three = np.where(og_test[:, 0] == label1)[0].tolist() + np.where(og_test[:, 0] == label2)[0].tolist()
    test_data = og_test[one_vs_three, :]
    # Split data
    X_train = train_data[:, 1:]
    y_train = train_data[:, 0]
    X_test = test_data[:, 1:]
    y_test = test_data[:, 0]
    # convert label to binary
    y_train_bi = y_train.copy()
    y_train_bi[y_train == label1 ] = -1
    y_train_bi[y_train == label2] = 1
    y_test_bi = y_test.copy()
    y_test_bi[y_test == label1 ] = -1
    y_test_bi[y_test == label2] = 1
    return X_train, y_train_bi, X_test, y_test_bi


def main_hw5():
    # Load data
    og_train_data = np.genfromtxt('zip.train')
    og_test_data = np.genfromtxt('zip.test')

    num_trees = 200

    # exp ones and threes
    print(" one_vs_three ")
    title = 'one_vs_three'
    X_train, y_train, X_test, y_test = subdata(1, 3, og_train_data, og_test_data, title = title)
    train_error, test_error = adaboost_trees(X_train, y_train, X_test, y_test, num_trees, title = title)
    print (" one_vs_three" , " train error=", train_error[-1], " test error=", test_error[-1])

    print (train_error[5])

    # exp threes and fives
    print(" three_vs_five ")
    title = 'three_vs_five'
    X_train, y_train, X_test, y_test = subdata(3, 5, og_train_data, og_test_data, title = title)
    train_error, test_error = adaboost_trees(X_train, y_train, X_test, y_test, num_trees, title = title)
    print (" three_vs_five" , " train error=", train_error[-1], " test error=", test_error[-1])


if __name__ == "__main__":
    main_hw5()
