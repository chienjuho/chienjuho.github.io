#!/usr/bin/python2.7
# Homework 1 Code
import numpy as np
import matplotlib.pyplot as plt


def perceptron_learn(data_in):
    # Run PLA on the input data
    #
    # Inputs: data_in: Assumed to be a matrix with each row representing an
    #                (x,y) pair, with the x vector augmented with an
    #                initial 1, and the label (y) in the last column
    # Outputs: w: A weight vector (should linearly separate the data if it is linearly separable)
    #        iterations: The number of iterations the algorithm ran for

    # Your code here, assign the proper values to w and iterations:

    (r, c) = data_in.shape

    # Extract X
    x = data_in[0:-1, :]

    # Extract Y
    y = data_in[-1, :].reshape(1, c)

    # Initialize w
    w = np.zeros((r - 1, 1))

    # h = sign(w'*x)
    h = np.sign(np.matmul(np.transpose(w), x))
    iterations = 0

    while not np.array_equal(h, y):
        # Obtain first element where h ~= y
        ind = np.where(np.not_equal(h, y))[1][0]

        w = w + (y[:, ind] * x[:, ind]).reshape(r - 1, 1)
        h = np.sign(np.matmul(np.transpose(w), x))

        iterations += 1

    return w, iterations


def perceptron_experiment(N, d, num_exp):
    # Code for running the perceptron experiment in HW1
    #
    # Inputs: N is the number of training examples
    #       d is the dimensionality of each example (before adding the 1)
    #       num_exp is the number of times to repeat the experiment
    # Outputs: num_iters is the # of iterations PLA takes for each sample
    #        bounds_minus_ni is the difference between the theoretical bound and the actual number of iterations
    # (both the outputs should be num_exp long)

    # Your code here, assign the values to num_iters and bounds_minus_ni:

    # Initialize the return variables
    num_iters = np.zeros((num_exp,))
    bounds_minus_ni = np.zeros((num_exp,))

    for i in range(num_exp):
        # Generate new training data with rand uniform of the numbers -1 and 1
        X = np.random.uniform(-1, 1, (d + 1, N))
        X[:, 0] = 1

        # Generate w and y
        w_star = np.vstack((np.array([0]), np.random.rand(d, 1)))

        y = np.sign(np.matmul(np.transpose(w_star), X))

        # Add y to training data
        training_data = np.vstack((X, y))

        # Run the algorithm, we now have w and the # of iterations for run

        _, iterations = perceptron_learn(training_data)

        # The theoretical run time, in accordance to part 1 of homework
        R = np.max(np.linalg.norm(X))
        rho = np.min(y * np.matmul(np.transpose(w_star), X))
        theoretical_bound = 1 / np.power(rho, 2) * (np.power(R, 2) * np.power(np.linalg.norm(w_star), 2))

        # Add number of iterations and the difference to return variables
        num_iters[i] = iterations
        bounds_minus_ni[i] = np.abs(theoretical_bound - iterations)

    return num_iters, bounds_minus_ni


def main():
    print("Running the experiment...")
    num_iters, bounds_minus_ni = perceptron_experiment(100, 10, 1000)

    print("Printing histogram...")
    plt.hist(num_iters)
    plt.title("Histogram of Number of Iterations")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Count")
    plt.show()

    print("Printing second histogram")
    plt.hist(np.log(bounds_minus_ni))
    plt.title("Bounds Minus Iterations")
    plt.xlabel("Log Difference of Theoretical Bounds and Actual # Iterations")
    plt.ylabel("Count")
    plt.show()


if __name__ == "__main__":
    main()
