from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from p_tqdm import p_map

from linear_regression import (
    compute_misclassification,
    linear_regression,
)
from perceptron import generate_data


def get_y(x: np.ndarray) -> int:
    """
    Implements the target function as:
        f(x1, x2) = sign(x1^2 + x2^2 - 0.6)

    Args:
        x: The input vector.

    Returns:
        The target value.
    """
    return np.sign(x[0] ** 2 + x[1] ** 2 - 0.6)


def run_experiment_one(N: int, runs: int = 1000, desired_num_of_plots: int = 5):
    """
    Runs the experiment for the first part of the homework.

    Args:
        N: The number of training examples.
        runs: The number of runs.
        desired_num_of_plots: The number of plots to generate.
    """

    def run_single_experiment_one(N: int, desired_num_of_plots: int, idx: int) -> float:
        np.random.seed(idx)
        a, b = 1, 0
        X_train, _ = generate_data(N, a, b)
        y_train = np.array([get_y(x) for x in X_train])
        w = linear_regression(X_train, y_train)
        ein = compute_misclassification(X_train, y_train, w)
        if idx < desired_num_of_plots:
            plt.figure()
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
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
            plt.title(f"N={N}, Ein={ein*100:.2f}%")
            plots_dir = "nlr_plots"
            (Path(__file__).parent / plots_dir).mkdir(exist_ok=True, parents=True)
            plt.savefig(Path(plots_dir) / f"nlr_idx_{idx+1}.png")
            plt.close()
        return ein

    eins = p_map(
        run_single_experiment_one,
        [N] * runs,
        [desired_num_of_plots] * runs,
        list(range(runs)),
    )
    return np.mean(eins)


def run_experiment_two(
    N_train: int, N_test: int, runs: int = 1000, desired_num_of_plots: int = 5
):
    """
    Runs the experiment for the second part of the homework.

    Args:
        N_train: The number of training examples.
        N_test: The number of test examples.
        runs: The number of runs.
        desired_num_of_plots: The number of plots to generate.
    """

    def run_single_experiment_two(
        N_train: int, N_test: int, desired_num_of_plots: int, idx: int
    ) -> float:
        np.random.seed(idx)
        a, b = 1, 0
        X_train, _ = generate_data(N_train, a, b)
        X_test, _ = generate_data(N_test, a, b)
        y_train = np.array([get_y(x) for x in X_train])
        y_test = np.array([get_y(x) for x in X_test])
        # Transform X from (x1, x2) to (1, x1, x2, x1x2, x1^2, x2^2)
        X_train_augmented = np.c_[
            np.ones(N_train), X_train, X_train[:, 0] * X_train[:, 1], X_train**2
        ]
        X_test_augmented = np.c_[
            np.ones(N_train), X_test, X_test[:, 0] * X_test[:, 1], X_test**2
        ]
        # Compute weights
        w = (
            np.linalg.inv(X_train_augmented.T @ X_train_augmented)
            @ X_train_augmented.T
            @ y_train
        )
        ein = compute_misclassification(X_train_augmented, y_train, w, augment=False)
        eout = compute_misclassification(X_test_augmented, y_test, w, augment=False)
        if idx < desired_num_of_plots:
            plt.figure()
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
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
            plt.title(f"N={N_train}, Ein={ein*100:.2f}%, Eout={eout*100:.2f}%")
            plots_dir = "nlr_exp2_plots"
            (Path(__file__).parent / plots_dir).mkdir(exist_ok=True, parents=True)
            plt.savefig(Path(plots_dir) / f"nlr_exp2_idx_{idx+1}.png")
            plt.close()
        return ein, eout, w

    eins, eouts, ws = zip(
        *p_map(
            run_single_experiment_two,
            [N_train] * runs,
            [N_test] * runs,
            [desired_num_of_plots] * runs,
            list(range(runs)),
        )
    )
    return np.mean(eins), np.mean(eouts), np.mean(ws, axis=0)


if __name__ == "__main__":
    N = 1000
    runs = 1000
    desired_num_of_plots = 5
    ein = run_experiment_one(N, runs, desired_num_of_plots)
    print(f"Average Ein: {ein*100:.2f}%")

    N_train = 1000
    N_test = 1000
    runs = 1000
    desired_num_of_plots = 5
    ein, eout, w = run_experiment_two(N_train, N_test, runs, desired_num_of_plots)
    print(f"Average Ein: {ein*100:.2f}%, Average Eout: {eout*100:.2f}%")
    print(f"Average weights: {w}")
