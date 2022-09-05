#!/usr/bin/python3
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

    # Extract X
    x =

    # Extract Y
    y =

    # Initialize w
    w =

    # h
    h =

    # updating
    while not np.array_equal(h, y):
        w =
        h =
        iterations += 1

    return w, iterations


def perceptron_experiment(N, d, num_samples):
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
    num_iters =
    bounds_minus_ni =

    for i in range(num_samples):
        # Generate new training data with rand uniform of the numbers -1 and 1
        X =

        # Generate w and y
        w_star =

        y =

        # Add y to training data
        training_data =

        # Run the algorithm, we now have w and the # of iterations for run

        _, iterations = perceptron_learn(training_data)

        # The theoretical run time, in accordance to part 1 of homework
        R =
        rho =
        theoretical_bound =

        # Add number of iterations and the difference to return variables
        num_iters[i] =
        bounds_minus_ni[i] =

    return num_iters, bounds_minus_ni


def main():
    print("Running the experiment...")
    num_iters, bounds_minus_ni = perceptron_experiment(100, 10, 1000)

    print("Printing histogram...")
    plt.hist(num_iters)
    plt.title("Histogram of Number of Iterations")
    plt.xlabel("Number of iterations")
    plt.show()

    print("Printing second histogram")
    plt.hist(np.log(bounds_minus_ni))
    plt.title("Bounds Minus Iterations")
    plt.xlabel("Log Difference")
    plt.show()


if __name__ == "__main__":
    main()
