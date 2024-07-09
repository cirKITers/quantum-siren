import itertools

import time
import plotly.graph_objects as go
import pandas as pd

import numpy as np

from matplotlib import pyplot as plt


def get_positive_spectrum_gaps(eigenvalues, unique=True):

    eigenvalues = list((sorted(eigenvalues, reverse=True)))

    gaps = []

    for i in range(0, len(eigenvalues) - 1):

        for j in range(i + 1, len(eigenvalues)):

            gap = eigenvalues[i] - eigenvalues[j]

            if gap > 1e-5:

                gaps.append(gap)

            else:

                print(f"Gap was too small {gap}")

    if unique:

        return list(set(gaps))

    else:

        return gaps


num_unique_gaps = []
num_unique_eigenvalues = []

# For simple feature maps, where the eigenvalues of the generator are qubit independent

max_n = 10

for n_qubits in range(1, max_n + 1):

    single_qubit_eigenvalues = [[-1 / 2.0, 1 / 2.0] for n in range(1, n_qubits + 1)]

    all_eigenvalues = [
        sum(element) for element in itertools.product(*single_qubit_eigenvalues)
    ]
    unique_eigenvalues = set(all_eigenvalues)

    gaps = get_positive_spectrum_gaps(unique_eigenvalues)

    print(gaps)

    num_unique_gaps.append(len(gaps))

    num_unique_eigenvalues.append(len(all_eigenvalues))


print(num_unique_gaps)


fig = go.Figure()

fig.add_trace(
    go.Scatter(
        # x=range(1, max_n + 1),
        y=num_unique_gaps,
        mode="lines+markers",
        name="# of Unique Gaps",
        marker=dict(
            size=10,
            color="#009682",
        ),
    )
)


fig.add_trace(
    go.Scatter(
        # x=range(1, max_n + 1),
        y=num_unique_eigenvalues,
        mode="lines+markers",
        name="# of Unique Eigenvalues",
        marker=dict(
            size=10,
            color="#4664AA",
        ),
    )
)

fig.update_layout(
    xaxis=dict(
        title="# of Qubits (n)",
    ),
    yaxis=dict(
        # title="Unique Eigenvalues",
        type="log",
    ),
    template="simple_white",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
)

fig.show()
