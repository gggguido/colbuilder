"""Benchmark corrected closure on saved pre-optimization copies, without Chimera."""

import argparse
import json
from pathlib import Path
import time
import warnings

import numpy as np
from Bio import BiopythonWarning

from colbuilder.core.sequence.optimize_crosslinks import load_pdb, save_pdb
from colbuilder.core.sequence.steric_optimization import (
    optimize_glucosepane,
    StericOptimizationError,
    get_residue,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--site", default="310.C - 542.B")
    parser.add_argument("--seed", type=int, default=1201)
    parser.add_argument("--input-pdb", type=Path)
    args = parser.parse_args()
    warnings.simplefilter("ignore", BiopythonWarning)
    args.output.mkdir(parents=True, exist_ok=False)
    np.random.seed(args.seed)
    pdb = args.input_pdb or next(args.source.glob("*+ADD1_Glucosepane_disoriented.pdb"))
    structures = {
        "initial": load_pdb(pdb),
        "copy1": load_pdb(args.source / "initial_copy1.pdb"),
        "copy2": load_pdb(args.source / "initial_copy2.pdb"),
    }
    endpoints = [p.strip().split(".") for p in args.site.split("-")]
    crosslink = {"R3": {"type": "NONE"}}
    for role, (number, chain), name, atom, sid in zip(
        ["R1", "R2"], endpoints, ["AGS", "LGX"], ["NZ", "CE"], ["copy1", "copy2"]
    ):
        crosslink[role] = dict(
            structure_id=sid, chain=chain, position=number, type=name, atom=atom
        )

    def forming_distance():
        xyz = [
            get_residue(structures[r["structure_id"]], r["chain"], r["position"])[
                r["atom"]
            ].coord
            for r in (crosslink["R1"], crosslink["R2"])
        ]
        return float(np.linalg.norm(xyz[0] - xyz[1]))

    direct = forming_distance()
    crosslink["R1"]["structure_id"], crosslink["R2"]["structure_id"] = "copy2", "copy1"
    if forming_distance() > direct:
        crosslink["R1"]["structure_id"], crosslink["R2"]["structure_id"] = (
            "copy1",
            "copy2",
        )
    before = {
        (c.id, r.id, a.name): a.coord.copy()
        for c in structures["initial"][0]
        for r in c
        for a in r
    }
    started = time.monotonic()
    result = dict(site=args.site, seed=args.seed)
    try:
        result.update(optimize_glucosepane(structures, crosslink))
        result["accepted"] = True
        result["max_backbone_displacement_A"] = max(
            float(np.linalg.norm(a.coord - before[(c.id, r.id, a.name)]))
            for c in structures["initial"][0]
            for r in c
            for a in r
            if a.name in {"N", "CA", "C", "O"}
        )
        for key, structure in structures.items():
            save_pdb(structure, str(args.output / (key + "_optimized.pdb")))
    except StericOptimizationError as exc:
        result.update(accepted=False, error=str(exc))
    result["elapsed_seconds"] = time.monotonic() - started
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
