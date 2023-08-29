import glob
import os
import yaml
import shutil

mlflow_path = "./mlruns/418881901150108894"
backup_dir = "./.mlruns_bckp"

cut_after = 1  # 1670237437014

runs = glob.glob(os.path.join(mlflow_path, "*"))

for r in runs:
    mark_for_deletion = False
    mark_for_deprecation = False
    if not os.path.isdir(r):
        continue
    with open(os.path.join(r, "meta.yaml"), "r") as f:
        try:
            content = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)

        if content["status"] != 3:
            mark_for_deletion = True
        elif int(content["end_time"]) < cut_after:
            mark_for_deprecation = True
            content["lifecycle_stage"] = "deleted"

    if mark_for_deletion:
        shutil.move(r, os.path.join(backup_dir, os.path.basename(r)))
    elif mark_for_deprecation:
        with open(os.path.join(r, "meta.yaml"), "w") as f:
            yaml.safe_dump(content, f)
