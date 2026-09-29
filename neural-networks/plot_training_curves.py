"""Summary figure: mean training accuracy of the MLP on XOR and on the 4-input M-of-N task.

    python plot_training_curves.py [output.png]

Uses the same settings as XOR.py and M-of-N.py (10 runs x 1,000 epochs, lr 0.1), averaged
over runs, with a fixed seed so the figure is reproducible.
"""
import itertools
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model import MLP

RUNS, EPOCHS, LR = 10, 1000, 0.1


def mean_accuracy(train, target, n_in, hidden, momentum, **kw):
    acc = np.zeros((RUNS, EPOCHS))
    for r in range(RUNS):
        mlp = MLP(n_in, 1, hidden, LR, momentum)
        _, acc[r], _ = mlp.train(train, target, EPOCHS, **kw)
    return acc.mean(axis=0)


def m_of_n(k):
    """Same dataset as M-of-N.py: label 1 when exactly k // 2 of the k inputs are on."""
    inputs = np.array(list(itertools.product([0, 1], repeat=k)))
    return inputs, np.array([[1] if sum(x) == k // 2 else [0] for x in inputs])


def main():
    np.random.seed(0)
    out = sys.argv[1] if len(sys.argv) > 1 else "training_curves.png"
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)

    xor_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    xor_y = np.array([[0], [1], [1], [0]])
    for m in [0.5, 0.9, 0.99]:
        a.plot(mean_accuracy(xor_x, xor_y, 2, [3], m, activation_type="sigmoid"), label=f"momentum {m}")
    a.set_title("XOR (hidden layer [3], sigmoid)")

    x, y = m_of_n(4)
    for hidden in [[5], [5, 5], [10, 10]]:
        b.plot(mean_accuracy(x, y, 4, hidden, 0.9), label=f"hidden {hidden}")
    b.set_title("4-input M-of-N (ReLU, momentum 0.9)")

    a.set_ylabel(f"Training accuracy (mean of {RUNS} runs)")
    for ax in (a, b):
        ax.set_xlabel("Epoch")
        ax.set_ylim(0.3, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


if __name__ == "__main__":
    main()
