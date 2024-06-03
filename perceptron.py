from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from p_tqdm import p_map


def target() -> Tuple[float, float]:
    """
    Generate a random target function f(x) = a * x + b

    Returns:
        Tuple[float, float]: a, b
    """
    # Generate two random points
    xs = np.random.uniform(-1, 1, (2, 2))
    # a = (y2 - y1) / (x2 - x1)
    a = (xs[1, 1] - xs[0, 1]) / (xs[1, 0] - xs[0, 0])
    # b = y1 - a * x1
    b = xs[0, 1] - a * xs[0, 0]
    return a, b


def generate_data(N: int, a: float, b: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate N random data points

    Args:
        N (int): Number of data points
        a (float): Slope of the target function
        b (float): Intercept of the target function

    Returns:
        Tuple[np.ndarray, np.ndarray]: X, y
    """
    X = np.random.uniform(-1, 1, (N, 2))
    y = np.sign(X[:, 1] - (a * X[:, 0] + b))
    return X, y


def pla(
    X: np.ndarray, y: np.ndarray, initial_weights: np.ndarray = None
) -> Tuple[np.ndarray, int]:
    """
    Perceptron Learning Algorithm

    Args:
        X (np.ndarray): Data points
        y (np.ndarray): Labels
        initial_weights (np.ndarray, optional): Initial weights. Defaults to None.

    Returns:
        Tuple[np.ndarray, int]: Weights, Number of iterations
    """
    # Start with w = 0
    if initial_weights is not None:
        w = initial_weights
    else:
        w = np.zeros(X.shape[1] + 1)
    # Add bias term
    X_augmented = np.c_[np.ones(X.shape[0]), X]
    # Initialize iterations
    iterations = 0
    while True:
        # Compute predictions
        y_pred = np.sign(np.dot(X_augmented, w))
        # Find misclassified points
        misclassified_points = np.where(y_pred != y)[0]
        # If no misclassified points, it has converged
        if len(misclassified_points) == 0:
            break
        # Else, pick a random misclassified point
        point = np.random.choice(misclassified_points)
        # Update weights
        w += y[point] * X_augmented[point]
        iterations += 1
    return w, iterations


def pocket_pla(
    X: np.ndarray, y: np.ndarray, initial_w: np.ndarray, max_iterations: int
) -> np.ndarray:
    """
    Pocket Perceptron Learning Algorithm

    Args:
        X (np.ndarray): Data points
        y (np.ndarray): Labels
        initial_w (np.ndarray): Initial weights
        max_iterations (int): Maximum number of iterations

    Returns:
        np.ndarray: Best weights
    """
    # Start with w = w0
    w = initial_w.copy()
    best_w = w.copy()
    # Initialize best error to infinity
    best_error = np.inf
    # Add bias term
    X_augmented = np.c_[np.ones(X.shape[0]), X]
    for _ in range(max_iterations):
        # Compute predictions
        y_pred = np.sign(np.dot(X_augmented, w))
        # Compute error
        misclassified_points = np.where(y_pred != y)[0]
        # If no misclassified points, it has converged
        if len(misclassified_points) == 0:
            break
        # Else, pick a random misclassified point
        point = np.random.choice(misclassified_points)
        # Update weights
        w += y[point] * X_augmented[point]
        # Compute error
        error = np.mean(np.sign(np.dot(X_augmented, w)) != y)
        # If error is less than best error, update best weights
        if error < best_error:
            best_error = error
            best_w = w.copy()
    return best_w


def estimate_misclassification_prob(
    a: float, b: float, w: np.ndarray, num_new_points: int = 10000
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate the probability of misclassification

    Args:
        a (float): Slope of the target function
        b (float): Intercept of the target function
        w (np.ndarray): Weights
        num_new_points (int, optional): Number of new points. Defaults to 10000.

    Returns:
        Tuple[float, np.ndarray, np.ndarray]: Misclassification probability, X, y
    """
    X, y = generate_data(num_new_points, a, b)
    X_augmented = np.c_[np.ones(X.shape[0]), X]
    y_pred = np.sign(np.dot(X_augmented, w))
    return np.mean(y_pred != y), X, y


def run_experiment(
    N: int, runs: int = 1000, desired_num_of_plots: int = 5
) -> Tuple[float, float]:
    """
    Run the experiment

    Args:
        N (int): Number of data points
        runs (int, optional): Number of runs. Defaults to 1000.
        desired_num_of_plots (int, optional): Number of plots to generate. Defaults to 5.

    Returns:
        Tuple[float, float]: Average iterations, Average misclassification probability
    """

    def run_single_experiment(
        N: int, desired_num_of_plots: int, idx: int
    ) -> Tuple[float, float]:
        np.random.seed(idx)
        a, b = target()
        X, y = generate_data(N, a, b)
        w, iter_count = pla(X, y)
        misclassifcation_prob, X_test, y_test = estimate_misclassification_prob(a, b, w)
        if idx < desired_num_of_plots:
            plt.figure()
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
            plt.plot([-1, 1], [a * -1 + b, a * 1 + b], label="Target Function")
            plt.plot(
                [-1, 1],
                [
                    (-w[0] - w[1] * -1) / w[2],
                    (-w[0] - w[1] * 1) / w[2],
                ],
                label="Learned Function",
            )
            plt.scatter(
                X_test[y_test == 1][:, 0],
                X_test[y_test == 1][:, 1],
                color="blue",
                s=1,
                label="y=1",
            )
            plt.scatter(
                X_test[y_test == -1][:, 0],
                X_test[y_test == -1][:, 1],
                color="red",
                s=1,
                label="y=-1",
            )
            plt.legend()
            plt.title(
                f"N={N}, Iterations={iter_count}, Misclassification Probability={misclassifcation_prob*100:.2f}%"
            )
            (Path(__file__).parent / "perceptron_plots").mkdir(
                exist_ok=True, parents=True
            )
            plt.savefig(Path("perceptron_plots") / f"perceptron_N_{N}_idx_{idx+1}.png")
            plt.close()
        return iter_count, misclassifcation_prob

    iterations, misclassification_probs = zip(
        *p_map(
            run_single_experiment,
            [N] * runs,
            [desired_num_of_plots] * runs,
            list(range(runs)),
        )
    )
    return np.mean(iterations), np.mean(misclassification_probs)


if __name__ == "__main__":
    iterations_N10, misclassification_prob_N10 = run_experiment(N=10, runs=1000)
    print(f"Average iterations for N=10: {iterations_N10}")
    print(
        f"Average misclassification probability for N=10: {misclassification_prob_N10}"
    )

    iterations_N100, misclassification_prob_N100 = run_experiment(N=100, runs=1000)
    print(f"Average iterations for N=100: {iterations_N100}")
    print(
        f"Average misclassification probability for N=100: {misclassification_prob_N100}"
    )
