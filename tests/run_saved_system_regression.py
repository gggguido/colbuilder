"""Rebuild an existing PYD/GCP/MOLD system from its actual active caps.

Optional AGE replacement runs through real Chimera. Work is done in a new
directory; old TOP includes identify active caps, not the new grouping.
"""

import argparse
import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import yaml

from colbuilder.core.geometry.crosslink_network import active_caps, resolve_network
from colbuilder.core.geometry.geometry_replacer import CrosslinkReplacer
from colbuilder.core.geometry.model import Model
from colbuilder.core.geometry.system import System
from colbuilder.core.topology.main_topology import build_topology
from colbuilder.core.utils.config import ColbuilderConfig


async def run(case, output, ratio, temporary_topology=False):
    output.mkdir(parents=True, exist_ok=False)
    topology = case / (".tmp/topology_gen" if temporary_topology else
                       "rattus_norvegicus_topology_files")
    source_top = topology / "collagen_fibril_rattus_norvegicus.top"
    groups = re.findall(r'#include\s+"col_([0-9_]+)\.itp"', source_top.read_text())
    geometry = output / ".tmp/geometry_gen"
    system = System()
    for group in groups:
        merged = list((case / ".tmp/topology_gen").glob(f"*/{group}.merge.pdb"))
        if len(merged) != 1:
            raise ValueError(f"Cannot identify original caps ownership for {group}: {merged}")
        kind = merged[0].parent.name
        for mid in map(int, group.split("_")):
            src = merged[0].parent / f"{mid}.caps.pdb"
            dest = geometry / kind / src.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            model = Model(mid, [0, 0, 0], connect=[mid], pdb_file=str(src))
            model.type = kind
            system.add_model(model)
    config_path = next(
        p
        for p in [case / "geometry_gen.yaml", case / "mix.yaml", case / "config.yaml"]
        if p.exists()
    )
    data = yaml.safe_load(config_path.read_text())
    data.update(
        working_directory=str(output),
        sequence_generator=False,
        geometry_generator=False,
        topology_generator=True,
        mix_bool=False,
        files_mix=None,
        auto_fix_unpaired=False,
        manual_replacements=None,
        replace_bool=ratio > 0,
        ratio_replace=ratio,
        ratio_replace_scope="non_enzymatic",
        debug=True,
        pdb_file=None,
        replace_file=None,
    )
    config = ColbuilderConfig(**data)
    initial = resolve_network(active_caps(system, geometry), config)
    before = Counter(e.kind for e in initial.entities)
    previous = Path.cwd()
    os.chdir(output)
    try:
        if ratio:
            await CrosslinkReplacer().replace_in_system(
                system, config, output / ".tmp/replace_crosslinks"
            )
        await build_topology(system, config)
        after = Counter(e.kind for e in system.crosslink_network.entities)
        assert after["PYD"] == before["PYD"]
        for kind in ("GCP", "MOLD"):
            assert after[kind] == before[kind] - round(before[kind] * ratio / 100)
        destination = output / "rattus_norvegicus_topology_files"
        mdp = output / "validation.mdp"
        mdp.write_text(
            "integrator=steep\nnsteps=0\nconstraints=none\ncutoff-scheme=Verlet\n"
            "nstlist=10\nverlet-buffer-tolerance=-1\nrlist=1.0\ncoulombtype=Cut-off\n"
            "rcoulomb=1.0\nvdwtype=Cut-off\nrvdw=1.0\npbc=xyz\n"
        )
        result = subprocess.run(
            [
                "gmx",
                "grompp",
                "-f",
                str(mdp),
                "-p",
                "collagen_fibril_rattus_norvegicus.top",
                "-c",
                "collagen_fibril_rattus_norvegicus.gro",
                "-o",
                str(output / "validation.tpr"),
                "-po",
                str(output / "mdout.mdp"),
            ],
            cwd=destination,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        (output / "grompp.log").write_text(result.stdout)
        assert result.returncode == 0, result.stdout[-3000:]
        report = {
            "case": str(case),
            "ratio_non_enzymatic": ratio,
            "before": before,
            "after": after,
            "groups": len(
                re.findall(r'#include\s+"col_', (destination / source_top.name).read_text())
            ),
            "grompp_returncode": result.returncode,
        }
        (output / "regression_result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        os.chdir(previous)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ratio", type=float, default=0)
    parser.add_argument("--temporary-topology", action="store_true",
                        help="Rebuild retained temporary inputs from a failed export.")
    args = parser.parse_args()
    asyncio.run(run(args.case.resolve(), args.output.resolve(), args.ratio,
                    args.temporary_topology))
