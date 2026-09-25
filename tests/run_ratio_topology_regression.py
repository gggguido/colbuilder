"""Run actual replacement + public topology pipeline on a fresh copy of saved caps.

Unlike the previous backbone replay this runs ratio selection and final group
construction too. It never reuses the old replacement list or group partition.
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

from colbuilder.core.geometry.backbone import read_pdb_residues
from colbuilder.core.geometry.crosslink_network import active_caps
from colbuilder.core.geometry.geometry_replacer import CrosslinkReplacer
from colbuilder.core.geometry.model import Model
from colbuilder.core.geometry.system import System
from colbuilder.core.topology.main_topology import build_topology
from colbuilder.core.utils.config import ColbuilderConfig


async def run(case, output, mode="random", seed=None):
    output.mkdir(parents=True, exist_ok=False)
    source_top = case / "rattus_norvegicus_topology_files/collagen_fibril_rattus_norvegicus.top"
    groups = re.findall(r'#include\s+"col_([0-9_]+)\.itp"', source_top.read_text())
    ids = sorted({int(mid) for group in groups for mid in group.split("_")})
    geometry = output / ".tmp/geometry_gen/T"
    geometry.mkdir(parents=True)
    system = System()
    for mid in ids:
        src = case / f".tmp/geometry_gen/T/{mid}.caps.pdb"
        shutil.copy2(src, geometry / src.name)
        model = Model(mid, [0, 0, 0], connect=[mid], pdb_file=str(src))
        model.type = "T"
        system.add_model(model)
    data = yaml.safe_load((case / "config.yaml").read_text())
    data.update(
        working_directory=str(output),
        sequence_generator=False,
        geometry_generator=False,
        topology_generator=True,
        mix_bool=False,
        files_mix=None,
        auto_fix_unpaired=False,
        manual_replacements=None,
        replace_bool=True,
        ratio_replace=50,
        ratio_replace_mode=mode,
        ratio_replace_seed=seed,
        debug=True,
    )
    (output / "regression_config.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    config = ColbuilderConfig(**data)
    previous = Path.cwd()
    os.chdir(output)
    try:
        replacement = output / ".tmp/replace_crosslinks"
        await CrosslinkReplacer().replace_in_system(system, config, replacement)
        assert system.crosslink_network.counts == {"PYD:enzymatic_c": 5, "PYD:enzymatic_n": 5}
        assert len((replacement / "replace.txt").read_text().splitlines()) == 30
        if mode == "preserve_attachment":
            from colbuilder.core.geometry.replacement_policy import validate_attachment_plan
            validate_attachment_plan(system.crosslink_network.replacement_report,
                                     system.crosslink_network.entities)
            assert not system.crosslink_network.replacement_report["attachment"]["newly_unlinked_models"]
        cap_paths = active_caps(system, replacement)
        for mid, after_path in cap_paths.items():
            before = read_pdb_residues(
                (geometry / after_path.name).read_text().splitlines(True), str(mid)
            )
            after = read_pdb_residues(after_path.read_text().splitlines(True), str(mid))
            assert [(r.address, r.segment) for r in before] == [
                (r.address, r.segment) for r in after
            ]
            for a, b in zip(before, after):
                for atom in {"N", "CA", "C", "O"} & a.atoms.keys():
                    assert a.atoms[atom][30:54] == b.atoms[atom][30:54]
        system.write_pdb(
            output / "collagen_fibril_rattus_norvegicus.pdb",
            67,
            cleanup=False,
            temp_dir=replacement,
        )
        await build_topology(system, config)
        topology = output / "rattus_norvegicus_topology_files"
        if mode == "preserve_attachment":
            final_check = json.loads((topology / "replacement_attachment_validation.json").read_text())
            assert final_check["status"] == "passed" and not final_check["newly_unlinked_models"]
            assert all(degree == 1 for degree in final_check["degree_from_itp_bonds"].values())
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
            cwd=topology,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        (output / "grompp.log").write_text(result.stdout)
        assert result.returncode == 0, result.stdout[-5000:]
        marker_counts = Counter(key[3] for key in system.crosslink_network.markers)
        report = {
            "source": str(case),
            "replacement_mode": mode,
            "replacement_seed": seed,
            "active_caps": len(ids),
            "remaining_entities": system.crosslink_network.counts,
            "marker_residues": marker_counts,
            "expected_crosslink_bonds": len(system.crosslink_network.entities) * 2,
            "grompp_returncode": result.returncode,
            "production_md": False,
        }
        (output / "regression_result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        os.chdir(previous)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["random", "preserve_attachment"], default="random")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    asyncio.run(run(args.case.resolve(), args.output.resolve(), args.mode, args.seed))
