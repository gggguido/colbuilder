"""Regression fixture: MOLD+GCP donor, AGS523.A / SER518.B, 2026-09-05."""
import json

import numpy as np
import pytest

from colbuilder.core.geometry.ring_validation import CapsGeometry, validate_caps_rings
from colbuilder.core.sequence.steric_optimization import GlucosepaneGeometry
from colbuilder.core.utils.ring_geometry import RingScreen, template_rings
from test_backbone_preservation import atom

AGS = {
    'N': (-.448, .352, -1.318), 'CA': (0., 0., 0.), 'CB': (1.496, -.332, .166),
    'CG': (1.827, -1.703, .760), 'CD': (3.338, -1.904, .924), 'NE': (3.570, -3.155, 1.699),
    'CZ': (4.789, -3.366, 2.280), 'NH1': (5.779, -2.435, 2.146), 'NH2': (5.017, -4.502, 3.),
    'C': (-.896, -1.069, .539), 'O': (-2.058, -1.180, .152), 'C15': (6.384, -4.379, 3.503),
    'C16': (6.807, -3.097, 2.859), 'C17': (7.790, -5.443, 1.788), 'C18': (9.073, -4.581, 1.716),
    'C19': (9.266, -3.579, 2.874), 'C20': (8.174, -2.521, 2.985), 'NZ': (7.194, -5.531, 3.129),
    'O18': (10.223, -5.448, 1.702), 'O19': (10.532, -2.921, 2.663),
}
SER = {'N': (4.979, -3.268, .867), 'CA': (4.423, -3.803, 2.078),
       'CB': (5.469, -4.472, 2.985), 'OG': (6.399, -3.504, 3.448),
       'C': (3.787, -2.703, 2.869), 'O': (3.784, -1.542, 2.462)}


def fixture_caps(tmp_path, split=False):
    lines = {0: []}
    if split:
        lines[1] = []
    for name, number, chain, xyz, mid in [('AGS', 523, 'A', AGS, 0), ('SER', 518, 'B', SER, int(split))]:
        for n, position in xyz.items():
            lines[mid].append(atom(len(lines[mid])+1, n, name, number, chain=chain, xyz=position))
        lines[mid].append('TER\n')
    caps = {}
    for mid, rows in lines.items():
        caps[mid] = tmp_path / f'{mid}.caps.pdb'
        caps[mid].write_text(''.join(rows) + 'END\n')
    return caps


@pytest.mark.parametrize('shift', [[0, 0, 0], [10000, -20000, 15000]])
def test_segment_surface_confirmed_and_external_and_same_side(shift):
    xyz = np.array([[-1,-1,0], [1,-1,0], [1,1,0], [-1,1,0], [.2,.1,-1], [.2,.1,1]], dtype=float) + shift
    screen = RingScreen(xyz, [(4,5)], [(0,1,2,3)])
    assert screen.check(xyz)[0]['status'] == 'penetration'
    outside = xyz.copy()
    outside[4:, 0] += 3
    assert not RingScreen(outside, [(4,5)], [(0,1,2,3)]).check(outside)
    same_side = xyz.copy()
    same_side[4:, 2] += 3
    assert not RingScreen(same_side, [(4,5)], [(0,1,2,3)]).check(same_side)


def test_incident_bonds_excluded_but_grazing_not_certified():
    xyz = np.array([[-1,-1,0], [1,-1,0], [1,1,0], [-1,1,0], [1,0,-1], [1,0,1]], dtype=float)
    hits = RingScreen(xyz, [(0,5), (4,5)], [(0,1,2,3)]).check(xyz)
    assert len(hits) == 1
    assert hits[0]['status'] == 'ambiguous'


def test_rtp_cycles_are_ordered_and_include_fused_ags_rings():
    rings = template_rings('AGS')
    assert sorted(map(len, rings)) == [5, 7]
    assert {'C15', 'C16', 'NH1', 'CZ', 'NH2'} in [set(r) for r in rings]


@pytest.mark.parametrize('split', [False, True])
def test_real_ser_penetration_before_mixing_detected_across_caps(tmp_path, split):
    caps = fixture_caps(tmp_path, split)
    original = {p: p.read_bytes() for p in caps.values()}
    report = validate_caps_rings(caps, repair=False)
    relevant = [h for h in report['before'] if {a['atom'] for a in h['bond_atoms']} == {'CB', 'OG'}]
    assert len(relevant) == 1 and relevant[0]['status'] == 'penetration'
    assert all(p.read_bytes() == content for p, content in original.items())


def test_repair_preserves_backbone_markers_and_native_bond_lengths(tmp_path):
    caps = fixture_caps(tmp_path)
    before = CapsGeometry(caps)
    original = before.coordinates.copy()
    report = validate_caps_rings(caps, repair=True, report_path=tmp_path / 'report.json')
    after = CapsGeometry(caps)
    assert report['penetrations_after'] == 0
    assert report['repairs']
    for i, label in enumerate(before.labels):
        if label['residue'].startswith('AGS') or label['atom'] in {'N','CA','C','O','CB'}:
            np.testing.assert_array_equal(after.coordinates[i], original[i])
    for a,b in before.bonds:
        assert np.linalg.norm(after.coordinates[a]-after.coordinates[b]) == pytest.approx(
            np.linalg.norm(original[a]-original[b]), abs=.002)
    assert caps[0].read_text().count('TER\n') == 2
    assert json.loads((tmp_path/'report.json').read_text())['penetrations_after'] == 0


def test_strict_policy_fails_without_writing_coordinates(tmp_path):
    caps = fixture_caps(tmp_path)
    before = caps[0].read_bytes()
    with pytest.raises(ValueError, match='unresolved'):
        validate_caps_rings(caps, policy='error', repair=False, report_path=tmp_path/'report.json')
    assert caps[0].read_bytes() == before
    assert json.loads((tmp_path/'report.json').read_text())['penetrations_after'] > 0


def test_sequence_rejects_threaded_candidate_even_with_finite_soft_score():
    result = dict(forming_A=1.5, minimum_nonbonded_A=2., minimum_14_A=2., score=1., ring_penetrations=1)
    assert not GlucosepaneGeometry.acceptable(result)


def test_moving_only_screen_matches_full_for_changed_ring_and_changed_bond():
    xyz = np.array([[-1,-1,0], [1,-1,0], [1,1,0], [-1,1,0], [3,0,-1], [3,0,1]], dtype=float)
    screen = RingScreen(xyz, [(4,5)], [(0,1,2,3)], moving=[4,5])
    new = xyz.copy()
    new[4:,0] = .1
    assert len(screen.check(new)) == len(RingScreen(new, [(4,5)], [(0,1,2,3)]).check(new)) == 1
    screen = RingScreen(xyz, [(4,5)], [(0,1,2,3)], moving=[0,1,2,3])
    new = xyz.copy()
    new[:4,0] += 3
    assert screen.check(new)[0]['status'] == 'penetration'


def test_nonfinite_coordinates_fail():
    with pytest.raises(ValueError, match='Nonfinite'):
        RingScreen([[float('nan'),0,0]], [], [])


def test_coplanar_contact_is_ambiguous_not_clear_or_confirmed():
    xyz = np.array([[-1,-1,0], [1,-1,0], [1,1,0], [-1,1,0], [-.5,0,0], [.5,0,0]], dtype=float)
    assert RingScreen(xyz, [(4,5)], [(0,1,2,3)]).check(xyz)[0]['status'] == 'ambiguous'


def test_actual_sequence_objective_uses_ring_screen():
    from Bio.PDB import Atom, Chain, Residue
    from test_glucosepane_steric_optimization import example

    structures, crosslink = example()
    for sid, structure in structures.items():
        shift = np.array([20., 0., 0.]) if sid == 'copy2' else np.zeros(3)
        target = structure[0]['C'][310]
        for name, xyz in AGS.items():
            target[name].coord = np.array(xyz) + shift
        chain = Chain.Chain('D')
        ser = Residue.Residue((' ', 518, ' '), 'SER', ' ')
        for i, (name, xyz) in enumerate(SER.items()):
            ser.add(Atom.Atom(name, np.array(xyz)+shift, 0, 1, ' ', name.rjust(4), i+1, element=name[0]))
        chain.add(ser)
        structure[0].add(chain)
    geometry = GlucosepaneGeometry(structures, crosslink)
    result = geometry.evaluate(np.zeros(geometry.dimension))
    assert result['ring_penetrations'] >= 2
    assert any(set(map(tuple, hit['bond_atoms'])) == {('copy1','D',518,'CB'), ('copy1','D',518,'OG')}
               for hit in result['ring_contacts'])
    assert not geometry.acceptable(result)


def test_partial_strict_repair_keeps_all_original_files(tmp_path):
    caps = fixture_caps(tmp_path)
    ring = np.array([AGS[n] for n in ('C15','C16','NH1','CZ','NH2')])
    center = ring.mean(axis=0)
    _,_,axes = np.linalg.svd(ring-center)
    path = tmp_path/'1.caps.pdb'
    path.write_text(atom(1,'N','GLY',999,chain='C',xyz=center-axes[-1])
                    + atom(2,'CA','GLY',999,chain='C',xyz=center+axes[-1]) + 'TER\nEND\n')
    caps[1] = path
    original = {p: p.read_bytes() for p in caps.values()}
    with pytest.raises(ValueError, match='unresolved'):
        validate_caps_rings(caps, policy='error', repair=True, report_path=tmp_path/'report.json')
    assert all(p.read_bytes() == data for p,data in original.items())


def test_detection_is_invariant_under_rigid_rotation(tmp_path):
    caps = fixture_caps(tmp_path)
    geometry = CapsGeometry(caps)
    original = geometry.screen().check(geometry.coordinates)
    q,_ = np.linalg.qr(np.array([[1.,2.,3.], [3.,1.,2.], [2.,3.,1.]]))
    moved = geometry.coordinates @ q + [50,30,-40]
    hits = RingScreen(moved, geometry.bonds, geometry.rings).check(moved)
    signature = lambda h: (tuple(h['ring']), tuple(h['bond']), h['status'])
    assert {signature(h) for h in hits} == {signature(h) for h in original}
