"""Opt-in real Chimera/GROMACS regression on a saved Colbuilder ratio run.

Run in the colbuilder environment. Inputs remain untouched; --output must be
a new directory. Replay the saved replacement selection to avoid random changes
in which crosslinks are removed. No geometry relaxation or MD is performed.
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
from types import SimpleNamespace

from colbuilder.core.geometry.backbone import (
    BackboneIntegrityError,
    preserve_backbone,
    read_pdb_residues,
    validate_backbone_topology,
)
from colbuilder.core.geometry.geometry_replacer import CrosslinkReplacer
from colbuilder.core.geometry.model import Model
from colbuilder.core.geometry.system import System
from colbuilder.core.topology.amber import Amber


def check_previous_topologies(case, output, groups):
    """Negative control: test the old ITPs against the intended polymer segments."""
    rejected = {}
    for group in groups:
        try:
            validate_backbone_topology(
                output / "T" / f"{group}.merge.pdb",
                case / "rattus_norvegicus_topology_files" / f"col_{group}.itp",
            )
        except BackboneIntegrityError as exc:
            rejected[group] = str(exc)
    return rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    case, output = args.case.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = case / ".tmp/geometry_gen/T"
    caps = output / "T"
    shutil.copytree(source, caps)
    instructions_path = output / "replace.txt"
    shutil.copy2(case / ".tmp/replace_crosslinks/replace.txt", instructions_path)
    instructions = instructions_path.read_text().splitlines()
    config = SimpleNamespace(
        CHIMERA_SCRIPTS_DIR=Path(__file__).resolve().parents[1] / "src/colbuilder/chimera_scripts",
        _replacement_verbose=True,
    )
    with preserve_backbone(caps, instructions):
        ok = asyncio.run(
            CrosslinkReplacer()._run_chimera_command(config, str(instructions_path), caps, output)
        )
        if not ok:
            raise RuntimeError("Chimera failed during the real regression")

    # Independently verify every cap's original boundaries and backbone positions.
    boundary_checks = 0
    for original in source.glob("*.caps.pdb"):
        before = read_pdb_residues(original.read_text().splitlines(keepends=True), str(original))
        after = read_pdb_residues(
            (caps / original.name).read_text().splitlines(keepends=True), original.name
        )
        assert [(r.address, r.segment) for r in before] == [(r.address, r.segment) for r in after]
        for a, b in zip(before, after):
            for atom in {"N", "CA", "C", "O"} & a.atoms.keys():
                assert a.atoms[atom][30:54] == b.atoms[atom][30:54]
        boundary_checks += 1

    topology = case / "rattus_norvegicus_topology_files"
    ff = "amber99sb-star-ildnp"
    shutil.copytree(topology / (ff + ".ff"), output / (ff + ".ff"))
    for filename in ("residuetypes.dat", "specbond.dat"):
        shutil.copy2(case / ".tmp/topology_gen" / filename, output / filename)
    original_top = topology / "collagen_fibril_rattus_norvegicus.top"
    groups = re.findall(r'#include\s+"col_([0-9_]+)\.itp"', original_top.read_text())
    system = System()
    for group in groups:
        members = [int(n) for n in group.split("_")]
        for mid in members:
            model = Model(
                float(mid),
                [0.0, 0.0, 0.0],
                connect=[float(n) for n in members],
                pdb_file=str(caps / f"{mid}.caps.pdb"),
            )
            model.type = "T"
            system.add_model(model)
    amber = Amber(system, ff=ff)
    env = {**os.environ, "GMXLIB": str(output)}
    previous = Path.cwd()
    os.chdir(output)
    try:
        processed = []
        for group in groups:
            kind, group_id, crosslinks = amber.merge_connected_models(
                [int(n) for n in group.split("_")]
            )
            pdb = output / kind / f"{group_id}.merge.pdb"
            command = [
                "gmx",
                "pdb2gmx",
                "-f",
                str(pdb),
                "-ignh",
                "-merge",
                "all",
                "-ff",
                ff,
                "-water",
                "tip3p",
                "-p",
                f"col_{group}.top",
                "-o",
                f"col_{group}.gro",
                "-i",
                f"posre_{group}.itp",
            ]
            result = subprocess.run(
                command,
                env=env,
                input="",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            (output / f"pdb2gmx_{group}.log").write_text(result.stdout)
            if result.returncode:
                raise RuntimeError(f"pdb2gmx failed for {group}; see its log")
            amber.write_itp(output / f"col_{group}.top", f"col_{group}", str(pdb), crosslinks)
            amber.ensure_posre_include(output / f"col_{group}.itp", group)
            processed.append((kind, group))
            print(f"Validated group {group}", flush=True)
        amber.write_topology("regression.top", processed)
        amber.write_gro("regression.gro", processed)
        (output / "validation.mdp").write_text(
            "; Diagnostic only, not a production MDP\nintegrator=steep\nnsteps=0\n"
            "constraints=none\ncutoff-scheme=Verlet\nnstlist=10\nverlet-buffer-tolerance=-1\n"
            "rlist=1.0\ncoulombtype=Cut-off\nrcoulomb=1.0\nvdwtype=Cut-off\nrvdw=1.0\npbc=xyz\n"
        )
        command = [
            "gmx",
            "grompp",
            "-f",
            "validation.mdp",
            "-p",
            "regression.top",
            "-c",
            "regression.gro",
            "-o",
            "validation.tpr",
            "-po",
            "mdout.mdp",
        ]
        result = subprocess.run(
            command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        (output / "grompp.log").write_text(result.stdout)
        if result.returncode:
            raise RuntimeError("grompp failed; see grompp.log")
        residues_count = Counter()
        for path in caps.glob("*.caps.pdb"):
            residues_count.update(
                r.name
                for r in read_pdb_residues(path.read_text().splitlines(keepends=True), str(path))
            )
        with (output / "regression.gro").open() as handle:
            handle.readline()
            atoms = int(handle.readline())
        report = {
            "source": str(case),
            "replacement_instructions": len(instructions),
            "caps_backbone_unchanged": boundary_checks,
            "groups_validated": len(processed),
            "atoms": atoms,
            "marker_residues": {
                k: residues_count[k] for k in ("LYX", "LY2", "LY3", "AGS", "LGX", "LZS", "LZD")
            },
            "grompp_returncode": result.returncode,
            "chainsep_override": False,
            "previous_topologies_rejected": check_previous_topologies(case, output, groups),
            "production_simulation": False,
        }
        (output / "regression_result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        os.chdir(previous)


if __name__ == "__main__":
    main()
