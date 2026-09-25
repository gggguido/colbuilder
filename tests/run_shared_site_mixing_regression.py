"""Validate a completed real shared-sites CLI run, without modifying its inputs.

Checks composed caps against their source backbones, final ledger and exported
ITPs. Runs grompp with no maxwarn override. Optionally performs a zero-step
force evaluation and an external independent graph audit.
"""

import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess

from colbuilder.core.geometry.backbone import (
    BACKBONE, read_pdb_residues, validate_backbone_topology,
)
from colbuilder.core.geometry.crosslink_network import resolve_network


def validate(case, output, evaluate_forces=False, independent_audit=None):
    output.mkdir(parents=True, exist_ok=False)
    mix = case / ".tmp/mixing_crosslinks"
    report = json.loads((mix / "crosslink_mix_report.json").read_text())
    assert report["strategy"] == "shared_sites"
    assert all(f["achieved"] == f["target"] for f in report["families"])
    caps = {int(mid): mix / "_mixed_sites" / f"{mid}.caps.pdb" for mid in report["model_sources"]}
    atom_checks = 0
    for mid, label in report["model_sources"].items():
        original = mix / label / f"{mid}.caps.pdb"
        final = caps[int(mid)]
        before = read_pdb_residues(original.read_text().splitlines(True), str(original))
        after = read_pdb_residues(final.read_text().splitlines(True), str(final))
        assert [(r.address, r.segment) for r in before] == [(r.address, r.segment) for r in after]
        for a, b in zip(before, after):
            for name in BACKBONE & a.atoms.keys():
                assert name in b.atoms and a.atoms[name][30:54] == b.atoms[name][30:54]
                atom_checks += 1
    network = resolve_network(caps)
    assert network.counts == report["counts"]
    ledger = json.loads((mix / "crosslink_network.json").read_text())
    identity = lambda entry: (entry["kind"], tuple(sorted(tuple(k) for k in entry["markers"])))
    assert {identity(e) for e in ledger["entities"]} == {e.identity for e in network.entities}
    assert {identity(e) for e in report["selection"]} == {e.identity for e in network.entities}

    topology = case / "rattus_norvegicus_topology_files"
    top = topology / "collagen_fibril_rattus_norvegicus.top"
    includes = re.findall(r'#include\s+"(col_[0-9_]+\.itp)"', top.read_text())
    assert includes
    for include in includes:
        group = include[len("col_"):-len(".itp")]
        merged = case / ".tmp/topology_gen/_mixed_sites" / f"{group}.merge.pdb"
        validate_backbone_topology(merged, topology / include)

    mdp = output / "validation.mdp"
    mdp.write_text(
        "integrator=steep\nnsteps=0\nconstraints=none\ncutoff-scheme=Verlet\n"
        "nstlist=10\nverlet-buffer-tolerance=-1\nrlist=1.0\ncoulombtype=Cut-off\n"
        "rcoulomb=1.0\nvdwtype=Cut-off\nrvdw=1.0\npbc=xyz\n"
    )
    def command(name, args, cwd):
        result = subprocess.run(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (output / f"{name}.log").write_text(result.stdout)
        assert result.returncode == 0, result.stdout[-4000:]
        return result
    command("grompp", ["gmx", "grompp", "-f", str(mdp), "-p", str(top),
                        "-c", str(topology / top.with_suffix(".gro").name),
                        "-o", str(output / "validation.tpr"), "-po", str(output / "mdout.mdp")], topology)
    result = {
        "counts": dict(Counter(e.kind for e in network.entities)),
        "alternative_families": report["families"], "models": len(caps),
        "backbone_atoms_unchanged": atom_checks, "backbone_itps_validated": len(includes),
        "largest_transfer_residual_A": max((t["max_residual_A"] for t in report["transfers"]), default=0),
        "grompp_returncode": 0,
    }
    if independent_audit:
        spec = importlib.util.spec_from_file_location("independent_graph_audit", independent_audit)
        audit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(audit)
        audit.OUT = output
        summary = audit.audit_case("shared_sites", case)
        result["independent_topology_audit"] = summary
        assert not summary["issues"], summary["issues"]
        assert not summary["duplicates"], summary["duplicates"]
    if evaluate_forces:
        # The exported GRO bounding box is not a solvated/production box.
        # Padding isolates the force diagnostic from periodic-image overlaps.
        command("editconf", ["gmx", "editconf", "-f", str(topology / top.with_suffix(".gro").name),
                              "-o", str(output / "vacuum.gro"), "-bt", "triclinic", "-d", "2.0"], output)
        command("grompp_padded", ["gmx", "grompp", "-f", str(mdp), "-p", str(top),
                                   "-c", str(output / "vacuum.gro"), "-o", str(output / "force_probe.tpr"),
                                   "-po", str(output / "mdout_padded.mdp")], topology)
        command("mdrun", ["gmx", "mdrun", "-s", "force_probe.tpr", "-deffnm", "zero_step",
                           "-nt", "1", "-nb", "cpu"], output)
        log = (output / "zero_step.log").read_text()
        result["force_diagnostic_box_padding_nm"] = 2.0
        finite = True
        for label in ("Potential Energy", "Maximum force", "Norm of force"):
            match = re.search(re.escape(label) + r"\s*=\s*(\S+)", log)
            value = float(match[1]) if match else float("nan")
            result[label] = value if math.isfinite(value) else str(value)
            finite = finite and math.isfinite(value)
        result["initial_forces_finite"] = finite
        result["MD_stability"] = "Not tested; zero-step finite forces are not minimization/equilibration"
    (output / "validation_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if evaluate_forces:
        assert result["initial_forces_finite"], "Non-finite initial forces: see validation_result.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluate-forces", action="store_true")
    parser.add_argument("--independent-audit", type=Path)
    args = parser.parse_args()
    validate(args.case.resolve(), args.output.resolve(), args.evaluate_forces, args.independent_audit)
