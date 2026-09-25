"""Exercise warning-only acceptance against saved positive/negative user cases."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import warnings

from Bio import BiopythonWarning
from Bio.PDB import PDBParser
import numpy as np
import yaml

from colbuilder.core.sequence.steric_optimization import GlucosepaneGeometry, get_residue
from colbuilder.core.topology.coordinate_validation import validate_glucosepane_contacts
from colbuilder.core.topology.crosslink_validation import sections
from colbuilder.core.utils.exceptions import TopologyGenerationError


def check_saved_gro(case):
    folder = case / 'rattus_norvegicus_topology_files'
    top = next(folder.glob('collagen_fibril_*.top'))
    itps = [folder / (f[0] + '.itp') for sec, f, _ in sections(top)
            if sec == 'molecules' for _ in range(int(f[1]))]
    try:
        return dict(accepted=True, metrics=validate_glucosepane_contacts(top.with_suffix('.gro'), itps))
    except TopologyGenerationError as exc:
        return dict(accepted=False, error=str(exc))


def inspect_sequence(folder, data, original):
    parser = PDBParser(QUIET=True)
    structures = {sid: parser.get_structure(sid, folder / '.tmp/sequence_gen' / f'optimized_{sid}.pdb')
                  for sid in ['copy1', 'copy2']}
    crosslink = {'R3': {'type': 'NONE'}}
    for key, part, name, atom, sid in zip(
        ['R1', 'R2'], data['additional_1_combination'].split('-'),
        ['AGS', 'LGX'], ['NZ', 'CE'], ['copy1', 'copy2'],
    ):
        number, chain = part.strip().split('.')
        crosslink[key] = dict(structure_id=sid, chain=chain, position=number, type=name, atom=atom)
    def distance():
        a, b = [get_residue(structures[r['structure_id']], r['chain'], r['position'])[r['atom']].coord
                for r in (crosslink['R1'], crosslink['R2'])]
        return float(np.linalg.norm(a - b))
    direct = distance()
    crosslink['R1']['structure_id'], crosslink['R2']['structure_id'] = 'copy2', 'copy1'
    if distance() > direct:
        crosslink['R1']['structure_id'], crosslink['R2']['structure_id'] = 'copy1', 'copy2'
    geometry = GlucosepaneGeometry(structures, crosslink)
    metrics = geometry.evaluate(np.zeros(geometry.dimension))
    metrics['accepted'] = geometry.acceptable(metrics)
    metrics['quality_warnings'] = geometry.quality_warnings(metrics)
    final = next(folder.glob('rattus*.pdb'))
    before, after = parser.get_structure('before', original), parser.get_structure('after', final)
    displacement = [float(np.linalg.norm(atom.coord - after[0][chain.id][residue.id][atom.name].coord))
                    for chain in before[0] for residue in chain for atom in residue
                    if atom.name in {'N', 'CA', 'C', 'O'}]
    metrics['max_backbone_displacement_A'] = max(displacement)
    metrics['pdb'] = str(final)
    return metrics


def main():
    warnings.simplefilter('ignore', BiopythonWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument('positive_case', type=Path)
    parser.add_argument('negative_case', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(positive_saved_GRO=check_saved_gro(args.positive_case),
                  negative_saved_GRO=check_saved_gro(args.negative_case))
    output = args.output / 'result.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    folder = args.output / 'sequence'
    folder.mkdir()
    data = yaml.safe_load((args.positive_case / 'non-enzymatic_seq.yaml').read_text())
    source = Path(data['mutated_pdb'])
    if not source.is_absolute():
        source = args.positive_case / source
    data.update(working_directory=str(folder), mutated_pdb=str(source), debug=True)
    config = folder / 'config.yaml'
    config.write_text(yaml.safe_dump(data, sort_keys=False))
    env = {**os.environ, 'PATH': str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH'],
           'PYTHONWARNINGS': 'ignore'}
    code = ('import random,numpy as np;random.seed(4201);np.random.seed(4201);'
            'from colbuilder.colbuilder import main;raise SystemExit(main(standalone_mode=False))')
    with (folder / 'cli.log').open('w') as log:
        process = subprocess.run([sys.executable, '-c', code, '--config_file', str(config)],
                                 cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
    result['sequence_returncode'] = process.returncode
    if process.returncode == 0:
        result['sequence'] = inspect_sequence(folder, data, source)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    passed = (result['positive_saved_GRO']['accepted'] and result['negative_saved_GRO']['accepted']
              and result['positive_saved_GRO']['metrics']['gross_overlaps'] == 0
              and result['negative_saved_GRO']['metrics']['gross_overlaps'] > 0
              and process.returncode == 0 and result['sequence']['accepted']
              and result['sequence']['max_backbone_displacement_A'] == 0)
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
