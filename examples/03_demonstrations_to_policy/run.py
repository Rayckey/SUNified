"""Train the small BC example and keep generated artifacts outside the checkout."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

from kuafu.paths import resolve_artifact_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="examples/bc-training")
    args = parser.parse_args()
    example = Path(__file__).resolve().parent
    output = resolve_artifact_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load((example / "input/train.yaml").read_text())
    dataset = output / "demonstrations.hdf5"
    shutil.copy2(example / "input/demonstrations.hdf5", dataset)
    config["dataset"]["dataset"] = str(dataset)
    config["logging"]["log_dir"] = str(output / "training")
    path = output / "train.yaml"
    path.write_text(yaml.safe_dump(config))
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MPLBACKEND": "Agg"}
    with (output / "train.log").open("w") as log:
        subprocess.run([sys.executable, "-m", "kuafu.bc.training.train_bc_prior_mlp", "--config", str(path)],
                       env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
