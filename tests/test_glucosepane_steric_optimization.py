import numpy as np
import pytest
from Bio.PDB import Atom, Chain, Model, Residue, Structure

from colbuilder.core.sequence.steric_optimization import (
    BACKBONE,
    GlucosepaneGeometry,
    MAX_FORMING_A,
    SidechainTorsions,
    StericOptimizationError,
    optimize_glucosepane,
    residue_templates,
)


def residue(name, number, seed):
    rng = np.random.default_rng(seed)
    result = Residue.Residue((" ", number, " "), name, " ")
    for i, atom_name in enumerate(residue_templates()[name]["atoms"]):
        element = atom_name.lstrip("0123456789")[0]
        atom = Atom.Atom(
            atom_name,
            rng.normal(size=3) * 3,
            0,
            1,
            " ",
            atom_name.rjust(4),
            i + 1,
            element=element,
        )
        result.add(atom)
    return result


@pytest.mark.parametrize("name", ["AGS", "LGX", "GLU"])
def test_torsions_preserve_every_internal_bond_angle_and_backbone(name):
    r = residue(name, 310, 17)
    template = residue_templates()[name]
    torsions = SidechainTorsions(r, template)
    assert torsions.rotations
    moved = torsions.coordinates(np.linspace(-2, 2, len(torsions.rotations)))
    adjacency = {n: set() for n in torsions.names}
    for a, b in template["bonds"]:
        if a not in adjacency or b not in adjacency:
            continue
        adjacency[a].add(b)
        adjacency[b].add(a)
        i, j = torsions.index[a], torsions.index[b]
        assert np.linalg.norm(moved[i] - moved[j]) == pytest.approx(
            np.linalg.norm(torsions.coords[i] - torsions.coords[j]), abs=1e-10
        )
    for center, neighbors in adjacency.items():
        for a in neighbors:
            for b in neighbors - {a}:
                i, j, k = [torsions.index[n] for n in (a, center, b)]

                def cosine(x):
                    v, w = x[i] - x[j], x[k] - x[j]
                    return np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w))

                assert cosine(moved) == pytest.approx(
                    cosine(torsions.coords), abs=1e-10
                )
    for name in BACKBONE & set(torsions.names):
        i = torsions.index[name]
        np.testing.assert_array_equal(moved[i], torsions.coords[i])


def test_glucosepane_rings_move_as_rigid_groups():
    torsions = SidechainTorsions(residue("AGS", 310, 17), residue_templates()["AGS"])
    names = {"CZ", "NH1", "NH2", "C15", "C16", "NZ", "C17", "C18", "C19", "C20"}
    indices = [torsions.index[n] for n in names]
    xyz = torsions.coordinates(np.ones(len(torsions.rotations)))
    before = torsions.coords[indices]
    after = xyz[indices]
    np.testing.assert_allclose(
        np.linalg.norm(before[:, None] - before, axis=2),
        np.linalg.norm(after[:, None] - after, axis=2),
        atol=1e-10,
    )


def example():
    structure = Structure.Structure("test")
    structure.add(Model.Model(0))
    for chain_id, name, number, seed in [
        ("A", "GLU", 312, 1),
        ("C", "AGS", 310, 2),
        ("B", "LGX", 542, 3),
    ]:
        chain = Chain.Chain(chain_id)
        chain.add(residue(name, number, seed))
        structure[0].add(chain)
    structures = {key: structure.copy() for key in ("initial", "copy1", "copy2")}
    for atom in structures["copy2"].get_atoms():
        atom.coord += np.array([20, 0, 0])
    crosslink = {
        "R1": {
            "structure_id": "copy1",
            "chain": "C",
            "position": "310",
            "type": "AGS",
            "atom": "NZ",
        },
        "R2": {
            "structure_id": "copy2",
            "chain": "B",
            "position": "542",
            "type": "LGX",
            "atom": "CE",
        },
        "R3": {"type": "NONE"},
    }
    return structures, crosslink


def test_contact_mask_comes_from_bonds_not_distance():
    structures, crosslink = example()
    geometry = GlucosepaneGeometry(structures, crosslink)
    ids = geometry.indices
    ne = ids[("copy1", "C", 310, "NE")]
    cb = ids[("copy1", "A", 312, "CB")]
    assert cb not in geometry.excluded[ne]
    nz, ce = geometry.ports
    assert ce in geometry.excluded[nz]
    assert ids[("copy2", "B", 542, "CG")] in geometry.pairs14[nz]


def test_bad_candidate_fails_without_mutating_inputs(monkeypatch):
    structures, crosslink = example()
    before = [a.coord.copy() for s in structures.values() for a in s.get_atoms()]
    monkeypatch.setattr(
        "colbuilder.core.sequence.steric_optimization.differential_evolution",
        lambda f, bounds, **kw: type("Result", (), {"x": np.zeros(len(bounds))})(),
    )
    with pytest.raises(StericOptimizationError, match="rejected"):
        optimize_glucosepane(structures, crosslink)
    after = [a.coord for s in structures.values() for a in s.get_atoms()]
    np.testing.assert_array_equal(before, after)


def test_apply_is_periodic_and_preserves_backbone():
    structures, crosslink = example()
    geometry = GlucosepaneGeometry(structures, crosslink)
    before = {
        (sid, c.id, r.id, a.name): a.coord.copy()
        for sid, s in structures.items()
        for c in s[0]
        for r in c
        for a in r
    }
    geometry.apply(np.full(geometry.dimension, 0.25))
    for sid, s in structures.items():
        for c in s[0]:
            for r in c:
                for a in r:
                    if a.name in BACKBONE or r.resname == "GLU":
                        np.testing.assert_array_equal(
                            a.coord, before[(sid, c.id, r.id, a.name)]
                        )
    for role in geometry.roles:
        for atom in structures["copy1"][0][role["chain"]][int(role["position"])]:
            translated = structures["copy2"][0][role["chain"]][int(role["position"])][
                atom.name
            ]
            np.testing.assert_allclose(
                translated.coord - atom.coord, [20, 0, 0], atol=1e-12
            )


def test_failed_replay_rolls_back_all_coordinates(monkeypatch):
    structures, crosslink = example()
    before = [a.coord.copy() for s in structures.values() for a in s.get_atoms()]
    answers = iter([True, False])
    monkeypatch.setattr(
        GlucosepaneGeometry, "acceptable", staticmethod(lambda result: next(answers))
    )
    monkeypatch.setattr(
        "colbuilder.core.sequence.steric_optimization.differential_evolution",
        lambda f, bounds, **kw: type("Result", (), {"x": np.full(len(bounds), 0.2)})(),
    )
    with pytest.raises(StericOptimizationError, match="replay validation failed"):
        optimize_glucosepane(structures, crosslink)
    np.testing.assert_array_equal(
        before, [a.coord for s in structures.values() for a in s.get_atoms()]
    )


def test_feasible_candidate_is_not_lost_to_an_invalid_final_iterate(monkeypatch):
    structures, crosslink = example()

    def evaluate(self, angles):
        return dict(
            score=1.0 if np.any(angles) else 2.0,
            forming_A=10.0 if np.any(angles) else 1.5,
            minimum_nonbonded_A=3.0,
            minimum_14_A=2.5,
        )

    def optimizer(f, bounds, **kwargs):
        f(np.zeros(len(bounds)))
        f(np.ones(len(bounds)))
        return type("Result", (), {"x": np.ones(len(bounds))})()

    monkeypatch.setattr(GlucosepaneGeometry, "evaluate", evaluate)
    monkeypatch.setattr(
        "colbuilder.core.sequence.steric_optimization.differential_evolution", optimizer
    )
    assert optimize_glucosepane(structures, crosslink)["forming_A"] == 1.5


@pytest.mark.parametrize(
    "field,value",
    [
        ("forming_A", MAX_FORMING_A + 0.01),
        ("forming_A", 0.8),
        ("minimum_nonbonded_A", 0.124),
        ("minimum_14_A", 0.24),
        ("minimum_nonbonded_A", float("nan")),
        ("minimum_14_A", float("nan")),
        ("forming_A", float("inf")),
        ("score", float("inf")),
        ("score", float("nan")),
    ],
)
def test_closure_alone_cannot_pass_validation(field, value):
    result = dict(forming_A=1.5, minimum_nonbonded_A=2.8, minimum_14_A=2.3, score=1)
    assert GlucosepaneGeometry.acceptable(result)
    result[field] = value
    assert not GlucosepaneGeometry.acceptable(result)


@pytest.mark.parametrize("forming", [1.5, 2.6, 3.3, MAX_FORMING_A])
@pytest.mark.parametrize("contact", [0.25, 0.323, 0.65, 1.49, 1.99])
def test_compressed_candidates_are_accepted_with_warnings(forming, contact):
    result = dict(forming_A=forming, minimum_nonbonded_A=contact,
                  minimum_14_A=0.323, score=10)
    assert GlucosepaneGeometry.acceptable(result) is True
    assert GlucosepaneGeometry.quality_warnings(result)


def test_warning_only_candidate_is_exported_without_extra_neighbor_moves(monkeypatch):
    structures, crosslink = example()
    monkeypatch.setattr(GlucosepaneGeometry, "evaluate", lambda self, angles: dict(
        forming_A=2.8, minimum_nonbonded_A=0.65, minimum_14_A=0.9, score=10))
    monkeypatch.setattr(
        "colbuilder.core.sequence.steric_optimization.differential_evolution",
        lambda f, bounds, **kw: type("Result", (), {"x": np.zeros(len(bounds))})(),
    )
    result = optimize_glucosepane(structures, crosslink)
    assert len(result["quality_warnings"]) == 3
    assert result["flexible_neighbors"] == []
