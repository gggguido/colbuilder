"""Re-run a user's two-stage GCP example in an isolated output directory."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--geometry", action="store_true")
    parser.add_argument("--third-site", default="310.C - 542.B")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.case / "rattusnorvegicus_N_PYD_C_PYD.pdb"
    results = []
    for step, filename in enumerate(
        ["1_non-enzymatic_seq.yaml", "2_non-enzymatic_seq.yaml"]
    ):
        dest = args.output / f"sequence_{step + 1}"
        dest.mkdir()
        data = yaml.safe_load((args.case / filename).read_text())
        data.update(working_directory=str(dest), mutated_pdb=str(source), debug=True)
        if step == 1:
            data["additional_1_combination"] = args.third_site
        config = dest / "config.yaml"
        config.write_text(yaml.safe_dump(data, sort_keys=False))
        code = (
            f"import random,numpy as np;random.seed({2201+step});np.random.seed({2201+step});"
            "from colbuilder.colbuilder import main;raise SystemExit(main(standalone_mode=False))"
        )
        env = {
            **os.environ,
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
            "PYTHONWARNINGS": "ignore",
        }
        with (dest / "cli.log").open("w") as log:
            run = subprocess.run(
                [sys.executable, "-c", code, "--config_file", str(config)],
                cwd=dest,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1800,
            )
        pdbs = list(dest.glob("rattus*.pdb"))
        result = dict(
            step=step + 1, returncode=run.returncode, pdbs=list(map(str, pdbs))
        )
        results.append(result)
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        if run.returncode != 0 or len(pdbs) != 1:
            raise SystemExit(
                "Sequence generation failed; inspect " + str(dest / "cli.log")
            )
        source = pdbs[0]
    if args.geometry:
        dest = args.output / "fibril"
        dest.mkdir()
        data = yaml.safe_load((args.case / "geometry_gen.yaml").read_text())
        data.update(working_directory=str(dest), pdb_file=str(source),
                    mutated_pdb=str(source), debug=True)
        config = dest / "config.yaml"
        config.write_text(yaml.safe_dump(data, sort_keys=False))
        with (dest / "cli.log").open("w") as log:
            run = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from colbuilder.colbuilder import main;raise SystemExit(main(standalone_mode=False))",
                    "--config_file",
                    str(config),
                ],
                cwd=dest,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1800,
            )
        results.append(dict(step="geometry/topology", returncode=run.returncode))
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results[-1]), flush=True)
        raise SystemExit(run.returncode)


if __name__ == "__main__":
    main()
