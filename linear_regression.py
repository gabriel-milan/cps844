from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from p_tqdm import p_map

from perceptron import generate_data, pla, pocket_pla, target


def compute_misclassification(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, augment: bool = True
) -> float:
    """
    Compute misclassification error

    Args:
        X (np.ndarray): Data points
        y (np.ndarray): Labels
        w (np.ndarray): Weights

    Returns:
        float: Misclassification error
    """
    # Add bias term
    if augment:
        X_augmented = np.c_[np.ones(X.shape[0]), X]
    else:
        X_augmented = X
    # Compute predictions
    y_pred = np.sign(np.dot(X_augmented, w))
    # Compute error
    error = np.mean(y_pred != y)
    return error


def invert_labels(y: np.ndarray, percentage: float):
    """
    Invert a percentage of the labels

    Args:
        y (np.ndarray): Labels
        percentage (float): Percentage of labels to invert

    Returns:
        np.ndarray: Inverted labels
    """
    # Number of labels to invert
    num_inverted = int(len(y) * percentage)
    # Randomly choose indices to invert
    indices = np.random.choice(len(y), num_inverted, replace=False)
    # Invert labels
    y[indices] = -y[indices]
    return y


def linear_regression(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Linear regression

    Args:
        X (np.ndarray): Data points
        y (np.ndarray): Labels

    Returns:
        np.ndarray: Weights
    """
    # Add bias term
    X_augmented = np.c_[np.ones(X.shape[0]), X]
    # Compute weights
    w = np.linalg.inv(X_augmented.T @ X_augmented) @ X_augmented.T @ y
    return w


def run_lr_experiment(
    N_train: int, N_test: int, runs: int = 1000, desired_num_of_plots: int = 5
):
    """
    Runs the linear regression experiment

    Args:
        N_train (int): Number of training data points
        N_test (int): Number of test data points
        runs (int, optional): Number of runs. Defaults to 1000.
        desired_num_of_plots (int, optional): Number of plots to generate. Defaults to 5.

    Returns:
        Tuple[float, float]: Ein, Eout
    """

    def run_single_experiment(
        N_train: int, N_test: int, idx: int, desired_num_of_plots: int
    ) -> Tuple[float, float]:
        np.random.seed(idx)
        a, b = target()
        X_train, y_train = generate_data(N_train, a, b)
        X_test, y_test = generate_data(N_test, a, b)
        w = linear_regression(X_train, y_train)
        ein = compute_misclassification(X_train, y_train, w)
        eout = compute_misclassification(X_test, y_test, w)
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
            plt.title(f"N_train={N_train}, Ein={ein*100:.2f}%, Eout={eout*100:.2f}%")
            plots_dir = "lr_plots"
            (Path(__file__).parent / plots_dir).mkdir(exist_ok=True, parents=True)
            plt.savefig(Path(plots_dir) / f"lr_idx_{idx+1}.png")
            plt.close()
        return ein, eout

    eins, eouts = zip(
        *p_map(
            run_single_experiment,
            [N_train] * runs,
            [N_test] * runs,
            list(range(runs)),
            [desired_num_of_plots] * runs,
        )
    )
    return np.mean(eins), np.mean(eouts)


def run_lr_pla_experiment(
    N_train: int, runs: int = 1000, desired_num_of_plots: int = 5
):
    """
    Runs the linear regression + PLA experiment

    Args:
        N_train (int): Number of training data points
        runs (int, optional): Number of runs. Defaults to 1000.
        desired_num_of_plots (int, optional): Number of plots to generate. Defaults to 5.

    Returns:
        Tuple[float, float]: Ein, Eout
    """

    def run_single_experiment(
        N_train: int, idx: int, desired_num_of_plots: int
    ) -> Tuple[float, float]:
        np.random.seed(idx)
        a, b = target()
        X_train, y_train = generate_data(N_train, a, b)
        w = linear_regression(X_train, y_train)
        _, iterations = pla(X_train, y_train, w)
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
                X_train[y_train == 1][:, 0],
                X_train[y_train == 1][:, 1],
                color="blue",
                s=1,
                label="y=1",
            )
            plt.scatter(
                X_train[y_train == -1][:, 0],
                X_train[y_train == -1][:, 1],
                color="red",
                s=1,
                label="y=-1",
            )
            plt.legend()
            plt.title(f"N_train={N_train}, Iterations={iterations}")
            plots_dir = "lr_pla_plots"
            (Path(__file__).parent / plots_dir).mkdir(exist_ok=True, parents=True)
            plt.savefig(Path(plots_dir) / f"lr_pla_idx_{idx+1}.png")
            plt.close()
        return iterations

    iterations = [
        *p_map(
            run_single_experiment,
            [N_train] * runs,
            list(range(runs)),
            [desired_num_of_plots] * runs,
        )
    ]
    return np.mean(iterations)


def run_pocket_pla_experiment(
    N_one: int,
    N_two: int,
    max_iterations: int,
    initialize_with_lr: bool,
    runs: int = 1000,
    desired_num_of_plots: int = 5,
) -> Tuple[float, float]:
    """
    Runs the pocket + PLA experiment

    Args:
        N_one (int): Number of data points for class 1
        N_two (int): Number of data points for class 2
        max_iterations (int): Maximum number of iterations for pocket PLA
        initialize_with_lr (bool): Whether to initialize the pocket PLA weights with linear
            regression results
        runs (int, optional): Number of runs. Defaults to 1000.
        desired_num_of_plots (int, optional): Number of plots to generate. Defaults to 5.

    Returns:
        Tuple[float, float]: Ein, Eout
    """

    def run_single_experiment(
        N_one: int,
        N_two: int,
        max_iterations: int,
        initialize_with_lr: bool,
        idx: int,
        desired_num_of_plots: int,
    ) -> Tuple[float, float]:
        np.random.seed(idx)
        a, b = target()
        X_train, y_train = generate_data(N_one, a, b)
        y_train = invert_labels(y_train, 0.1)
        X_test, y_test = generate_data(N_two, a, b)
        if initialize_with_lr:
            w = linear_regression(X_train, y_train)
        else:
            w = np.zeros(X_train.shape[1] + 1)
        w = pocket_pla(X_train, y_train, w, max_iterations)
        ein = compute_misclassification(X_train, y_train, w)
        eout = compute_misclassification(X_test, y_test, w)
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
                f"N1={N_one}, N2={N_two}, max_iterations={max_iterations}, initialize_with_lr={initialize_with_lr}, Ein={ein*100:.2f}%, Eout={eout*100:.2f}%"
            )
            plots_dir = "pocket_pla_plots"
            (Path(__file__).parent / plots_dir).mkdir(exist_ok=True, parents=True)
            plt.savefig(
                Path(plots_dir)
                / f"pocket_pla_init_{initialize_with_lr}_maxiter_{max_iterations}_idx_{idx+1}.png"
            )
            plt.close()
        return ein, eout

    eins, eouts = zip(
        *p_map(
            run_single_experiment,
            [N_one] * runs,
            [N_two] * runs,
            [max_iterations] * runs,
            [initialize_with_lr] * runs,
            list(range(runs)),
            [desired_num_of_plots] * runs,
        )
    )
    return np.mean(eins), np.mean(eouts)


if __name__ == "__main__":
    N_train = 100
    N_test = 1000
    ein, eout = run_lr_experiment(N_train, N_test)
    print("--- Linear Regression ---")
    print(f"Average Ein for N=100: {ein}")
    print(f"Average Eout for N=100: {eout}")

    N_train = 10
    iterations = run_lr_pla_experiment(N_train)
    print("--- Linear Regression + PLA ---")
    print(f"Average iterations for N=10: {iterations}")

    configs = [
        {
            "N_one": 100,
            "N_two": 1000,
            "max_iterations": 10,
            "initialize_with_lr": False,
        },
        {
            "N_one": 100,
            "N_two": 1000,
            "max_iterations": 50,
            "initialize_with_lr": False,
        },
        {"N_one": 100, "N_two": 1000, "max_iterations": 10, "initialize_with_lr": True},
        {"N_one": 100, "N_two": 1000, "max_iterations": 50, "initialize_with_lr": True},
    ]
    for config in configs:
        ein, eout = run_pocket_pla_experiment(**config)
        print(
            f"--- Pocket PLA (N1={config['N_one']}, N2={config['N_two']}, max_iterations={config['max_iterations']}, initialize_with_lr={config['initialize_with_lr']}) ---"
        )
        print(f"Average Ein: {ein}")
        print(f"Average Eout: {eout}")
