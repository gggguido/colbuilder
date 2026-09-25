"""Run the user's complete YAML in a fresh directory and retain its CLI log."""

import argparse
import json
from pathlib import Path
import subprocess

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.config.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    data = yaml.safe_load(source.read_text())
    data.update(working_directory=str(output), files_mix=None, debug=True)
    config = output / "config.yaml"
    config.write_text(yaml.safe_dump(data, sort_keys=False))
    command = ["colbuilder", "--config_file", str(config)]
    with (output / "cli.log").open("w") as log:
        result = subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT)
    report = {"source_config": str(source), "command": command, "returncode": result.returncode}
    topology = output / f"{data['species']}_topology_files"
    expected = [
        topology / f"collagen_fibril_{data['species']}.top",
        topology / f"collagen_fibril_{data['species']}.gro",
        topology / "crosslink_network.json",
    ]
    report["outputs_exist"] = all(path.is_file() for path in expected)
    (output / "cli_result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if result.returncode or not report["outputs_exist"]:
        raise SystemExit(result.returncode or 1)


if __name__ == "__main__":
    main()
