"""Exercise ring repair on saved full caps without changing their source files."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from colbuilder.core.geometry.backbone import BACKBONE
from colbuilder.core.geometry.crosslink_network import MARKER_NAMES, resolve_network
from colbuilder.core.geometry.ring_validation import CapsGeometry, validate_caps_rings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--policy', choices=['warn', 'error'], default='error')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    caps = {}
    for path in sorted(args.source.glob('*.caps.pdb')):
        target = args.output / path.name
        shutil.copy2(path, target)
        caps[int(path.name.split('.')[0])] = target
    assert caps
    network = resolve_network(caps)
    before = CapsGeometry(caps, network)
    report = validate_caps_rings(caps, network, repair=True, policy=args.policy,
                                 report_path=args.output/'ring_geometry_report.json')
    after = CapsGeometry(caps, network)
    assert before.labels == after.labels
    delta = np.linalg.norm(after.coordinates-before.coordinates, axis=1)
    fixed = [i for i, key in enumerate(before.atom_residues)
             if before.labels[i]['atom'] in BACKBONE or before.residues[key].name in MARKER_NAMES]
    assert not np.any(delta[fixed])
    bonds = np.array(before.bonds)
    d0 = np.linalg.norm(before.coordinates[bonds[:,0]]-before.coordinates[bonds[:,1]], axis=1)
    d1 = np.linalg.norm(after.coordinates[bonds[:,0]]-after.coordinates[bonds[:,1]], axis=1)
    assert np.max(abs(d1-d0)) < .004
    assert {e.identity for e in resolve_network(caps).entities} == {e.identity for e in network.entities}
    result = dict(models=len(caps), counts=network.counts, fixed_atoms=len(fixed),
                  moved_heavy_atoms=int(np.count_nonzero(delta)), max_displacement_A=float(delta.max()),
                  max_bond_length_change_A=float(np.max(abs(d1-d0))),
                  penetrations_before=report['penetrations_before'], penetrations_after=report['penetrations_after'])
    (args.output/'regression_result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
