import mlflow
import argparse
import re
import json
import plotly.graph_objects as go
import pandas as pd
import hashlib

# parser = argparse.ArgumentParser()
# parser.add_argument("--uri", type=str, default="")
# args = parser.parse_args()

input_str = 'http://localhost:5000/#/compare-runs?runs=["f8ace49a350446c097185f9b23c3de1a","5df81dbe41224f8d852aa2e66d2a6e65","b44e4a83bcf4447e88417a09c457e597","f6835b998b54453e80a9c1aad9a0484b","4bc6b7f43b4a4a28b4e42b85b39fee78","322b153d18cf42958b49396ed08459cd","57c435ac0d19480893a5b6c6cf159377","e79ebf1ba0eb45fd859c07aeaaa2c63b","1a369056e2ef427e820486c4c7c2fbc1","3442563cd5d14a6686392b4ae14f4a40","3f6306b6c90343dfa945491e476679f5","b20b2f3f807145d4ae16ae9d79eafd1a","683d490900cd4f4da01e3f8d2d68e624","d0089197f7ac442985e335d4e90ae8ec","0e5329ab727e4249b68ff9e1b4797986","5a26a558817240309ad0d92d22fdfbcf","28f14d5a457f494abad6f9bf54d8bdb2","063ab206d76b4678b84bf86d7fd045cb","dd1b637864a04a78b2b11b67c4194416","e379f338cbba4949ba5a81de28500aea","f983e3a98fe0464284a5eca8b206ee40","fb0122f472c74b01adcb982fb0915bf6","e63249b70b504912976715497b765ded","6c69174ac7d749998a142e385c624323","c8f0763754dc4662910364876dbebdac","2fa64f9896d041bb90b06c101d2c3994","35f8251d021749398ee26000d283ce6d","5e4de449e17648e1942bc28b1b12b284","3c5029ce75e0466aacadd16480df0b28","b48ee9b88cc0407aa48145ee45a4f68d","3d5c91411114481abbd6be49ef635a77","0a63fb0b708a42daac21f7b41166fd38","4a98f143769c4b9d9442ba545e581849","417b1891c82b46f8be891ba011c5bf5f","3331cd792f0a4f919b91507e7d6d3fc7","f25b32dfd3bf42fba1987e09f5150e99"]&experiments=["342195376402529587"]'

pattern = r"runs=(?P<runs>.*)&experiments=(?P<experiments>.*)"
match = re.search(pattern, input_str)

run_ids = json.loads(match["runs"])
experiment_ids = json.loads(match["experiments"])


run_id_condition = "'" + "','".join(run_ids) + "'"

complex_filter = f"""
attributes.run_id IN ({run_id_condition})
"""

runs = mlflow.search_runs(
    experiment_ids=experiment_ids,
    filter_string=complex_filter,
)
print(runs)

runs["params.training.n_layers"] = pd.to_numeric(
    runs["params.training.n_layers"]
).astype("Int64")
runs["params.training.n_qubits"] = pd.to_numeric(
    runs["params.training.n_qubits"]
).astype("Int64")

runs.sort_values(["params.training.n_layers", "params.training.n_qubits"], inplace=True)

fig = go.Figure(
    go.Contour(
        z=runs["metrics.mse"],
        x=runs["params.training.n_layers"],
        y=runs["params.training.n_qubits"],
        colorscale="PuRd",
        contours_coloring="heatmap",
        line_smoothing=0.85,
    )
)
fig.update_layout(yaxis_title="# Qubits", xaxis_title="# Layers")

hs = hashlib.md5(input_str.encode("utf-8")).hexdigest()
fig.write_image(f"mse-{hs}.png", scale=3)
