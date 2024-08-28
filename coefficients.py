from qml_essentials.model import Model
from functools import partial
from pennylane.fourier import coefficients
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from rich.progress import track

ansatz = (
    "Hardware_Efficient"  # Circuit_19,Circuit_18,Strongly_Entangling,Hardware_Efficient
)


def calculate_coefficients(n_qubits, n_layers, noise_factor=0.01):
    model = Model(
        n_qubits=n_qubits,
        n_layers=n_layers,
        circuit_type=ansatz,
    )

    model.noise_params = {
        "BitFlip": noise_factor,
        "PhaseFlip": noise_factor,
        "AmplitudeDamping": noise_factor,
        "PhaseDamping": noise_factor,
        "DepolarizingChannel": noise_factor,
    }

    partial_circuit = partial(model, model.params, noise_params=model.noise_params)
    # print(model.degree)
    return coefficients(partial_circuit, 1, model.degree)


max_n = 7
n_qubits = 3
samples = 200
max_noise = 0.03


def sample_coefficients(n_qubits, n_layers, noise_factor=0.01):
    coeffs_pl = np.ndarray((max_n, samples), dtype=complex)

    for n in range(n_layers):
        print(f"Complexity {n+1} of {n_layers}")
        for s in track(range(samples)):

            coeffs = calculate_coefficients(n_qubits, n + 1, noise_factor)

            coeff_z = coeffs[0]
            coeffs_nz = coeffs[1:]
            coeffs_p = coeffs_nz[len(coeffs_nz) // 2 :]
            coeffs_pl[n][s] = coeffs_p[-1]
            # print(coeffs_p.var(axis=0))

    coeffs_plr = coeffs_pl.real
    coeffs_pli = coeffs_pl.imag

    return coeffs_plr, coeffs_pli


fig = go.Figure()


# coeffs_plr, coeffs_pli = sample_coefficients(n_qubits, max_n, 0.01)

# fig.add_trace(
#     go.Scatter(
#         x=np.arange(1, max_n + 1),
#         y=coeffs_plr.var(axis=1),
#         mode="lines+markers",
#         name="Real Part",
#         marker=dict(
#             size=10,
#             color="#4664AA",
#         ),
#     )
# )

# fig.add_trace(
#     go.Scatter(
#         x=np.arange(1, max_n + 1),
#         y=coeffs_pli.var(axis=1),
#         mode="lines+markers",
#         name="Imaginary Part",
#         marker=dict(
#             size=10,
#             color="#009682",
#         ),
#     )
# )

noise_steps = 3
colors = ["#009682", "#4664AA", "#A22223"]
it = 0
for noise in np.arange(0.0, max_noise, max_noise / noise_steps):
    print(f"Calculating for step {it+1}/{noise_steps}")
    coeffs_plr, coeffs_pli = sample_coefficients(n_qubits, max_n, noise)
    coeffs_abs = np.sqrt(coeffs_plr**2 + coeffs_pli**2)

    fig.add_trace(
        go.Scatter(
            x=np.arange(1, max_n + 1),
            y=coeffs_abs.var(axis=1),
            mode="lines+markers",
            name=f"Noise Level: {noise:.2f}",
            marker=dict(
                size=10,
                color=colors[it],
            ),
        )
    )

    it += 1

fig.update_layout(
    xaxis_title="# of Unique Frequencies",
    yaxis_title="Variance of HF Coefficient (abs)",
    title=f"Variance of Coefficients for {ansatz}",
    template="simple_white",
    legend=dict(yanchor="top", y=1.0, xanchor="left", x=0.71),
)
fig.update_yaxes(type="log")
fig.show()
fig.write_image("coefficients.png")
# print(coeffs_plr.var(axis=1))
input()
