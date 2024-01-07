#!/usr/bin/python3
# Homework 4 Code
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import mode
import numpy as np
import matplotlib.pyplot as plt


def random_forest(X_train, y_train, X_test, y_test, num_bags, m):
    # The `random_forest` function learns an ensemble of numBags CART decision trees using a random subset of the features
    # at each split on the input dataset and also plots the  out-of-bag error as a function of the number of bags
    #
    # % Inputs:
    # % * `X_train` is the training data
    # % * `y_train` are the training labels
    # % * `X_test` is the testing data
    # % * `y_test` are the testing labels
    # % * `num_bags` is the number of trees to learn in the ensemble
    # % * `m` is the number of randomly selected features to consider at each split
    #
    # % Outputs:
    # % * `out_of_bag_error` is the out-of-bag classification error of the final learned ensemble
    # % * `test_error` is the classification error of the final learned ensemble on test data
    #
    # % Note: You may use sklearns 'DecisonTreeClassifier'
    # but **not** 'RandomForestClassifier' or any other bagging function

    out_of_bag_predictions = np.empty([X_train.shape[0], num_bags])
    out_of_bag_predictions[:] = np.nan
    test_predictions = np.empty([X_test.shape[0], num_bags])
    test_predictions[:] = np.nan

    out_of_bag_errors = []

    for i in range(num_bags):
        in_bag_indicies = np.random.choice(X_train.shape[0], X_train.shape[0])
        out_of_bag_indicies = np.setdiff1d(np.arange(0, X_train.shape[0]), in_bag_indicies)

        X_in_bag = X_train[in_bag_indicies, :]
        y_in_bag = y_train[in_bag_indicies]

        X_out_bag = X_train[out_of_bag_indicies, :]
        y_out_bag = y_train[out_of_bag_indicies]

        decision_tree = DecisionTreeClassifier(criterion="entropy")
        # print(X_in_bag, y_in_bag)
        decision_tree.fit(X_in_bag, y_in_bag)

        out_of_bag_predictions[out_of_bag_indicies, i] = decision_tree.predict(X_out_bag)
        test_predictions[:, i] = decision_tree.predict(X_test)

        aggregate_out_of_bag_predictions = mode(out_of_bag_predictions[:, 0:i + 1], axis=1)[0]
        valid_predictions = ~np.isnan(aggregate_out_of_bag_predictions).reshape(-1)

        out_of_bag_errors.append(np.sum(
            aggregate_out_of_bag_predictions[valid_predictions].reshape(-1) != np.array(
                y_train[valid_predictions])) / len(
            y_train[valid_predictions]))

    plt.plot(np.arange(1, num_bags+1), out_of_bag_errors)
    plt.xlabel('Number of bags')
    plt.ylabel('Out-of-bag error')
    plt.title('Random Forest')
    plt.show()

    out_of_bag_error = out_of_bag_errors[-1]
    aggregate_test_predictions = mode(test_predictions, axis=1)[0]
    test_error = np.sum(aggregate_test_predictions.reshape(-1) != y_test) / X_test.shape[0]

    return out_of_bag_error, test_error


def cross_val_decision_tree(X_train, y_train, X_test, y_test, m, cv):
    # The `cross_val_decision_tree` function select best decision tree by cross validation
    #
    # % Inputs:
    # % * `X_train` is the training data
    # % * `y_train` are the training labels
    # % * `X_test` is the testing data
    # % * `y_test` are the testing labels
    # % * `m` is the number of randomly selected features to consider at each split
    # % * `cv` is the number of number of folds for cross validation
    #
    # % Outputs:
    # % * `cv_error` is the cross validation error of the best decision tree
    # % * `test_error` is the classification error of the best decision tree on test data

    depths = range(2,21)
    cv_scores_list = []
    for d in depths:
        tree_model = DecisionTreeClassifier(max_features=m, max_depth=d)
        # tree_model = DecisionTreeClassifier(max_features=d)
        cv_scores = cross_val_score(tree_model, X_train, y_train, cv=cv, scoring='accuracy')
        cv_scores_list.append(cv_scores.mean())
    d_idx = np.argmax(np.array(cv_scores_list))
    d_max = depths[d_idx]

    best_tree = DecisionTreeClassifier(max_features=m, max_depth=d_max)
    # best_tree = DecisionTreeClassifier(max_features=d_max)
    best_tree.fit(X_train, y_train)
    test_prediction = best_tree.predict(X_test)
    test_error = np.sum(test_prediction != y_test) / X_test.shape[0]
    cv_error = 1 - cv_scores_list[d_idx]

    return  cv_error, test_error


def main_hw4(num_bags=200):
    print("number of bags is ", num_bags)

    # Load data
    og_train_data = np.genfromtxt('zip.train')
    og_test_data = np.genfromtxt('zip.test')

    # Find ones and threes
    print(" one_vs_three ")
    one_vs_three = np.where(og_train_data[:, 0] == 1)[0].tolist() + np.where(og_train_data[:, 0] == 3)[0].tolist()
    train_data = og_train_data[one_vs_three, :]
    one_vs_three = np.where(og_test_data[:, 0] == 1)[0].tolist() + np.where(og_test_data[:, 0] == 3)[0].tolist()
    test_data = og_test_data[one_vs_three, :]

    # Split data
    X_train = train_data[:, 1:]
    y_train = train_data[:, 0]
    X_test = test_data[:, 1:]
    y_test = test_data[:, 0]

    # Set parameters
    m = int(np.floor(X_train.shape[1] / 3))
    cv = 10

    # Run random forest
    out_of_bag_errors = []
    test_errors = []
    for i in range(0, 1):
        out_of_bag_error, test_error = random_forest(X_train, y_train, X_test, y_test, num_bags, m)
        out_of_bag_errors.append(out_of_bag_error)
        test_errors.append(test_error)

    print('The range of out-of-bag error for the random forest is between %.4f and %.4f' % (
        np.min(out_of_bag_errors), np.max(out_of_bag_errors)))
    print('The range of test error for the random forest is between %.4f and %.4f' % (
        np.min(test_errors), np.max(test_errors)))

    cross_val_score, test_error = cross_val_decision_tree(X_train, y_train, X_test, y_test, m, cv)
    print("cross val error of decision tree is ", cross_val_score)
    print("test error of decision tree is ", test_error)

    # Find threes and fives
    print(" three_vs_five ")
    five_vs_three = np.where(og_train_data[:, 0] == 5)[0].tolist() + np.where(og_train_data[:, 0] == 3)[0].tolist()
    train_data = og_train_data[five_vs_three, :]
    five_vs_three = np.where(og_test_data[:, 0] == 5)[0].tolist() + np.where(og_test_data[:, 0] == 3)[0].tolist()
    test_data = og_test_data[five_vs_three, :]

    # Split data
    X_train = train_data[:, 1:]
    y_train = train_data[:, 0]
    X_test = test_data[:, 1:]
    y_test = test_data[:, 0]

    # Set parameters
    m = int(np.floor(X_train.shape[1] / 3))

    # Run random forest
    out_of_bag_errors = []
    test_errors = []
    for i in range(0, 1):
        out_of_bag_error, test_error = random_forest(X_train, y_train, X_test, y_test, num_bags, m)
        out_of_bag_errors.append(out_of_bag_error)
        test_errors.append(test_error)

    print('The range of out-of-bag error for the random forest is between %.4f and %.4f' % (
        np.min(out_of_bag_errors), np.max(out_of_bag_errors)))
    print('The range of test error for the random forest is between %.4f and %.4f' % (
        np.min(test_errors), np.max(test_errors)))

    cross_val_score, test_error = cross_val_decision_tree(X_train, y_train, X_test, y_test, m, cv)
    print("cross val error of decision tree is ", cross_val_score)
    print("test error of decision tree is ", test_error)


def main_cv():
    # Load data
    og_train_data = np.genfromtxt('zip.train')
    og_test_data = np.genfromtxt('zip.test')

    # Find ones and threes
    print(" one_vs_three ")
    one_vs_three = np.where(og_train_data[:, 0] == 1)[0].tolist() + np.where(og_train_data[:, 0] == 3)[0].tolist()
    train_data = og_train_data[one_vs_three, :]
    one_vs_three = np.where(og_test_data[:, 0] == 1)[0].tolist() + np.where(og_test_data[:, 0] == 3)[0].tolist()
    test_data = og_test_data[one_vs_three, :]

    # Split data
    X_train = train_data[:, 1:]
    y_train = train_data[:, 0]
    X_test = test_data[:, 1:]
    y_test = test_data[:, 0]

    # Set parameters
    m = int(np.floor(X_train.shape[1] / 3))
    cv = 10
    cross_val_score, test_error = cross_val_decision_tree(X_train, y_train, X_test, y_test, m, cv)
    print("cross val error of decision tree is ", 1 - cross_val_score)
    print("test error of decision tree is ", test_error)


if __name__ == "__main__":
    np.random.seed(153)
    # num_bags = 1
    # main_hw4(num_bags)
    num_bags = 200
    main_hw4(num_bags)
    # main_cv()
