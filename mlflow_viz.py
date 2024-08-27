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

# input_str = 'http://localhost:5000/#/compare-runs?runs=["f8ace49a350446c097185f9b23c3de1a","5df81dbe41224f8d852aa2e66d2a6e65","b44e4a83bcf4447e88417a09c457e597","f6835b998b54453e80a9c1aad9a0484b","4bc6b7f43b4a4a28b4e42b85b39fee78","322b153d18cf42958b49396ed08459cd","57c435ac0d19480893a5b6c6cf159377","e79ebf1ba0eb45fd859c07aeaaa2c63b","1a369056e2ef427e820486c4c7c2fbc1","3442563cd5d14a6686392b4ae14f4a40","3f6306b6c90343dfa945491e476679f5","b20b2f3f807145d4ae16ae9d79eafd1a","683d490900cd4f4da01e3f8d2d68e624","d0089197f7ac442985e335d4e90ae8ec","0e5329ab727e4249b68ff9e1b4797986","5a26a558817240309ad0d92d22fdfbcf","28f14d5a457f494abad6f9bf54d8bdb2","063ab206d76b4678b84bf86d7fd045cb","dd1b637864a04a78b2b11b67c4194416","e379f338cbba4949ba5a81de28500aea","f983e3a98fe0464284a5eca8b206ee40","fb0122f472c74b01adcb982fb0915bf6","e63249b70b504912976715497b765ded","6c69174ac7d749998a142e385c624323","c8f0763754dc4662910364876dbebdac","2fa64f9896d041bb90b06c101d2c3994","35f8251d021749398ee26000d283ce6d","5e4de449e17648e1942bc28b1b12b284","3c5029ce75e0466aacadd16480df0b28","b48ee9b88cc0407aa48145ee45a4f68d","3d5c91411114481abbd6be49ef635a77","0a63fb0b708a42daac21f7b41166fd38","4a98f143769c4b9d9442ba545e581849","417b1891c82b46f8be891ba011c5bf5f","3331cd792f0a4f919b91507e7d6d3fc7","f25b32dfd3bf42fba1987e09f5150e99"]&experiments=["342195376402529587"]'

experiments = [
    # 1d cosine
    {
        "uri": 'http://localhost:5000/#/compare-runs?runs=["a7b8dedb3b61406790838300de23cc5d","dc49888dbbad4208ac3fb116efb6e359","e603bd2b5c874539a6789674857afaf9","e823b76323764a739b6f9b9dddfcbf36","aea3722629dc4b89ad36d6fc3eabcf16","6916bd68e1cd4202a1ebd9ceb2223fda","449c0fa22ea24b2997b43cd9bf61ceaf","7707ae0cf2754042b810f6c84563c3d3","3625d057fe23481daf6c0eab1ddac015","b8dffb54f47b418db8b448a986be96ce","da08e86a89a7476aa0c302041295f9a7","75f18711b8d144d6ac40377bd4b6a62f","89dea94727444d3c9d28c330b07ef2f0","b05d092752bd491b816dba3bfdf52669","371f9a580d8342c0a1ec8e6a73466ef5","905c1d432b0c4bbd8aef676ba136e229","010b1491634c4522a73162eb7928ef41","b42c2c227b4845839a44277e8ab8dbcf","f80b3447f0f64320ad5cf1bc3a7b6e61","50f9095a90c54991a854cabd160e7fcc","3ea9663672ee465ab40b614f5a9a4d41","5859e7109ef74e9a93a22fd0bdb7212c","882e095ca760478aab8b07353821024a","9a7f6940555c4ec48e620fe73fb238a4","9032ce91a2404d919e7566a27732c54a","10cd07e73c694329b7805482a132a288","cbbeb6ed72634fc9be23f40af85e0d25","55c2566c7af04ddebc9b9a9aa14eac2b","a7e2e4331c1948b6ab676752ea5f7e4f","7f7b03d42fb741ad9154702d2af0eff1","bf63a131e9f84c00ab23124bfd40f304","68b90ad981c14ccb9619522a4b700fa6","e49fa0452c084267939df60ebede19d3","4d4b240250f648338522ae72029e6792","35b4f994abeb4ca2a60bb2015b7597e7"]&experiments=["342195376402529587"]',
        "ssim": False,
    },
    # camera man
    {
        "uri": 'http://localhost:5000/#/compare-runs?runs=["15b9a2c1401048319a7c1d1b7654a3e0","340834da6f174f4499dd718ff0d8df35","89543aeb7bab4584a8f5a8e32c6741d0","afa18269fd394ec0a8a556a9871fb5dd","c57b0707fefa48ef9e94312387efa336","5f1df79496ff47e89d2a5e84770e8a7a","0b49c744e23140a48df912f136ad5506","5314e0af02a74736a01f5de35a7524eb","3df49fee6c0243faaef69c6e5fdb93ea","d5f7a3890fa24ad5b10bf7bb98c57bfb","aa6bb883eb894c3ca58a8fb24f5edb31","738b6630ea1349a8a98318b9dc0c988b","0915ef87268d4344a81059b85fbc987c","0670f100a70b49dfa7748b745aed7f48","1a389f13e3174f529397263c91c8b6dc","440929ae72184cbd8e37dd4ff7cfc6e0","83571019d60e496db9832c358bb06332","8d80264cfa40410ba572b17a9f781bf8","8ea4348bb0924986a890da96c0c81553","e852a9e0e7c44b5483a0d931bc365311","f0c95639f1a7463d89139f6531ef1851","102c753a37794e9682ae2e79de17671e","138521ef91044b01ab6ea88b96896453","177e6f207fb34267a95dfe92083f72f2","257c2b45dab94060adc91f438f94bda6","3711301634514194800c7c5902b21366","3d87e73e969b42298d289d25fbd3b7da","52b2daecb0aa475b90101c050105128b","55d88941b4374184b7fc6b44bf7c9d1c","98e69258e5504fc2821c4596683642fc","a7a730ce9cff413982c5292078df92c0","aa0db66383424d74a909c3921e3c2430","bb0aa8bc79fe4a3398dce757493c264a","c66d21187ca546e5b6b052de1f2d5f66","ce408761e51c4df99daf2713acf13716","d64ece40e826460f804e24412587e7e2","e0a467b2a10f4a54bbb5fee28ca3fc9c","f2b490a4a29c4da6b48cbf1fb1a2403c","f750279a891e4a218c6a915b0bf8dee3","0edbf43ce3fc4eb99fffebe6a10118c0","b72203a9032d403bba15a88a8a201af8","d21a0f667bc74cd2afd4e8c93b73a26d"]&experiments=["342195376402529587"]',
        "ssim": True,
    },
    # 2d cosine
    {
        "uri": 'http://localhost:5000/#/compare-runs?runs=["f8ace49a350446c097185f9b23c3de1a","5df81dbe41224f8d852aa2e66d2a6e65","b44e4a83bcf4447e88417a09c457e597","f6835b998b54453e80a9c1aad9a0484b","4bc6b7f43b4a4a28b4e42b85b39fee78","322b153d18cf42958b49396ed08459cd","57c435ac0d19480893a5b6c6cf159377","e79ebf1ba0eb45fd859c07aeaaa2c63b","1a369056e2ef427e820486c4c7c2fbc1","3442563cd5d14a6686392b4ae14f4a40","3f6306b6c90343dfa945491e476679f5","b20b2f3f807145d4ae16ae9d79eafd1a","683d490900cd4f4da01e3f8d2d68e624","d0089197f7ac442985e335d4e90ae8ec","0e5329ab727e4249b68ff9e1b4797986","5a26a558817240309ad0d92d22fdfbcf","28f14d5a457f494abad6f9bf54d8bdb2","063ab206d76b4678b84bf86d7fd045cb","dd1b637864a04a78b2b11b67c4194416","e379f338cbba4949ba5a81de28500aea","f983e3a98fe0464284a5eca8b206ee40","fb0122f472c74b01adcb982fb0915bf6","e63249b70b504912976715497b765ded","6c69174ac7d749998a142e385c624323","c8f0763754dc4662910364876dbebdac","2fa64f9896d041bb90b06c101d2c3994","35f8251d021749398ee26000d283ce6d","5e4de449e17648e1942bc28b1b12b284","3c5029ce75e0466aacadd16480df0b28","b48ee9b88cc0407aa48145ee45a4f68d","3d5c91411114481abbd6be49ef635a77","0a63fb0b708a42daac21f7b41166fd38","4a98f143769c4b9d9442ba545e581849","417b1891c82b46f8be891ba011c5bf5f","3331cd792f0a4f919b91507e7d6d3fc7","f25b32dfd3bf42fba1987e09f5150e99"]&experiments=["342195376402529587"]',
        "ssim": True,
    },
]
for experiment in experiments:

    pattern = r"runs=(?P<runs>.*)&experiments=(?P<experiments>.*)"
    match = re.search(pattern, experiment["uri"])

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

    runs["params.training.n_layers"] = pd.to_numeric(
        runs["params.training.n_layers"]
    ).astype("Int64")
    runs["params.training.n_qubits"] = pd.to_numeric(
        runs["params.training.n_qubits"]
    ).astype("Int64")

    runs.sort_values(
        ["params.training.n_layers", "params.training.n_qubits"], inplace=True
    )

    fig = go.Figure(
        go.Contour(
            z=(runs["metrics.mse"] if not experiment["ssim"] else runs["metrics.ssim"]),
            x=runs["params.training.n_layers"],
            y=runs["params.training.n_qubits"],
            contours_coloring="heatmap",  # heatmap
            # line_smoothing=0.85,
            colorscale=(
                "Bluyl" if not experiment["ssim"] else "Bluyl_r"
            ),  # Electric_r Bluyl PuRd
            contours=dict(
                showlabels=True,
                labelfont=dict(  # label font properties
                    size=20,
                    color="#A22223",
                ),
            ),
        )
    )
    fig.update_layout(
        # yaxis_title="# Qubits",
        # xaxis_title="# Layers",
        font=dict(
            size=22,
        ),
        template="simple_white",
        width=1200,
        height=400,
        margin={"t": 0, "l": 0, "b": 0, "r": 0},
    )

    hs = hashlib.md5(experiment["uri"].encode("utf-8")).hexdigest()
    fig.write_image(f"mse-{hs}.png", scale=3)
