from types import SimpleNamespace

import pytest

from colbuilder.core.geometry.crosslink_network import (
    CrosslinkIntegrityError,
    Entity,
    Marker,
    Network,
    active_caps,
    after_replacement,
    install_network,
    refresh_network,
    resolve_network,
    select_replacements,
    replacement_survivors,
)
from colbuilder.core.geometry.geometry_replacer import CrosslinkReplacer
from colbuilder.core.geometry.connect import LATTICE_GROWTH_CUTOFF
from test_backbone_preservation import atom


def terminal_network():
    markers, entities = {}, []
    for terminal, number in [("n", "944"), ("c", "103")]:
        for i in range(10):
            keys = (
                (i * 2, number, "B", "LYX"),
                (i * 2 + 1, number, "A", "LY2"),
                (i * 2 + 1, number, "C", "LY3"),
            )
            markers.update((k, Marker(k, {})) for k in keys)
            entities.append(
                Entity(
                    "PYD",
                    f"enzymatic_{terminal}",
                    keys,
                    (
                        (keys[0] + ("C12",), keys[1] + ("CB",)),
                        (keys[0] + ("C13",), keys[2] + ("CG",)),
                    ),
                )
            )
    return Network(markers, tuple(entities))


@pytest.mark.parametrize("seed", range(100))
def test_ratio_counts_unique_complete_crosslinks_per_terminal(seed):
    network = terminal_network()
    instructions, report = select_replacements(network, 50, "enzymatic", seed)
    assert len(instructions) == len(set(instructions)) == 30
    assert report["removed_entities"] == 10
    assert all(
        t == {"available": 10, "removed": 5, "remaining": 5} for t in report["targets"].values()
    )
    selected = {(int(f.split()[0].split(".")[0]), f.split()[2], f.split()[3]) for f in instructions}
    assert all(sum(key[:3] in selected for key in e.markers) in (0, 3) for e in network.entities)


@pytest.mark.parametrize("ratio,expected", [(0, 0), (20, 12), (30, 18), (50, 30), (100, 60)])
def test_ratio_edges_and_no_global_rng_changes(ratio, expected):
    import random

    state = random.getstate()
    assert len(select_replacements(terminal_network(), ratio, "enzymatic", 7)[0]) == expected
    assert random.getstate() == state


@pytest.mark.parametrize("ratio", [-1, 101, float("nan"), float("inf")])
def test_invalid_ratios_fail(ratio):
    with pytest.raises(CrosslinkIntegrityError):
        select_replacements(terminal_network(), ratio, "all", 1)


def test_connect_file_reverse_and_repeated_rows_are_deduplicated(tmp_path):
    path = tmp_path / "connect.txt"
    path.write_text(
        "1.caps.pdb 2.caps.pdb ; T\n2.caps.pdb 1.caps.pdb ; T\n"
        "1.caps.pdb 2.caps.pdb ; T\n3.caps.pdb ; T\n"
    )
    assert CrosslinkReplacer()._load_connect_groups(path) == [[1, 2], [3]]


def write_markers(path, records):
    lines, serial = [], 0
    for residue, number, chain, coordinates in records:
        for name, xyz in coordinates.items():
            serial += 1
            lines.append(atom(serial, name, residue, number, chain=chain, xyz=xyz))
        lines.append("TER\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines) + "END\n")


def long_pyd(tmp_path):
    a, b = tmp_path / "T/0.caps.pdb", tmp_path / "T/1.caps.pdb"
    write_markers(a, [("LYX", 944, "B", {"C12": (0, 0, 0), "C13": (0, 1, 0), "CA": (3, 0, 0)})])
    write_markers(
        b,
        [
            ("LY2", 5, "B", {"CB": (7.4, 0, 0), "CA": (4, 0, 0)}),
            ("LY3", 9, "C", {"CG": (6.4, 1, 0), "CA": (4, 1, 0)}),
        ],
    )
    cfg = SimpleNamespace(n_term_combination="9.C - 5.B - 944.B", c_term_combination=None)
    return {0: a, 1: b}, cfg


def test_long_pyd_requires_named_unambiguous_recipe(tmp_path):
    caps, cfg = long_pyd(tmp_path)
    network = resolve_network(caps, cfg)
    assert network.counts == {"PYD:enzymatic_n": 1}
    assert len(network.pairs([0, 1])) == 2
    with pytest.raises(CrosslinkIntegrityError, match="unpaired"):
        resolve_network(caps)
    with pytest.raises(CrosslinkIntegrityError, match="split"):
        network.pairs([0])


def test_recipe_does_not_join_remote_unrelated_caps(tmp_path):
    caps, cfg = long_pyd(tmp_path)
    text = caps[1].read_text()
    lines = []
    for line in text.splitlines(keepends=True):
        if line.startswith("ATOM"):
            line = line[:30] + f"{float(line[30:38])+100:8.3f}" + line[38:]
        lines.append(line)
    caps[1].write_text("".join(lines))
    with pytest.raises(CrosslinkIntegrityError, match="unpaired"):
        resolve_network(caps, cfg)


def test_partial_replacement_and_marker_inventory_changes_are_rejected(tmp_path):
    caps, cfg = long_pyd(tmp_path)
    network = resolve_network(caps, cfg)
    with pytest.raises(CrosslinkIntegrityError, match="Partial"):
        after_replacement(network, ["0.caps.pdb LYS 944 B"], caps)
    caps[0].write_text(caps[0].read_text().replace("LYX", "LYS"))
    with pytest.raises(CrosslinkIntegrityError, match="inventory"):
        refresh_network(network, caps)


def test_manual_lowercase_chain_removes_whole_entity(tmp_path):
    caps, cfg = long_pyd(tmp_path)
    network = resolve_network(caps, cfg)
    retained, markers = replacement_survivors(
        network, ["0.caps.pdb LYS 944 b", "1.caps.pdb LYS 5 b", "1.caps.pdb LYS 9 c"]
    )
    assert retained == [] and markers == set()


def test_duplicate_network_cannot_inflate_ratio_pool():
    network = terminal_network()
    network.entities += (network.entities[0],)
    with pytest.raises(CrosslinkIntegrityError, match="Duplicate"):
        select_replacements(network, 50, "enzymatic", 1)


def test_missing_forming_atoms_can_be_cleaned_manually_but_not_exported(tmp_path):
    caps, cfg = long_pyd(tmp_path)
    caps[0].write_text(
        "".join(
            line for line in caps[0].read_text().splitlines(True) if line[12:16].strip() != "C12"
        )
    )
    network = resolve_network(caps, cfg, strict=False)
    assert not network.entities
    with pytest.raises(CrosslinkIntegrityError, match="unpaired"):
        resolve_network(caps, cfg)


def test_same_final_graph_in_memory_and_on_disk(tmp_path):
    caps, cfg = long_pyd(tmp_path)
    network = resolve_network(caps, cfg)
    models = {i: SimpleNamespace(type="T", connect=[i]) for i in caps}
    system = SimpleNamespace(get_models=lambda: list(models), get_model=lambda mid: models[mid])
    assert install_network(system, network, caps, tmp_path) == [[0, 1]]
    assert models[0].connect == models[1].connect == [0, 1]
    assert (tmp_path / "connect_from_colbuilder.txt").read_text() == "0.caps.pdb 1.caps.pdb ; T\n"
    assert LATTICE_GROWTH_CUTOFF == 3.0
    assert active_caps(system, tmp_path) == caps


@pytest.mark.parametrize(
    "family,left,right,atom_left,atom_right",
    [
        ("GCP", "LGX", "AGS", "CE", "NZ"),
        ("MOLD", "LZD", "LZS", "CE", "NZ1"),
    ],
)
def test_age_scope_and_native_residues(tmp_path, family, left, right, atom_left, atom_right):
    caps = {0: tmp_path / "0.caps.pdb", 1: tmp_path / "1.caps.pdb"}
    write_markers(caps[0], [(left, 300, "A", {atom_left: (0, 0, 0)})])
    write_markers(caps[1], [(right, 500, "B", {atom_right: (1.5, 0, 0)})])
    network = resolve_network(caps)
    assert network.counts == {f"{family}:non_enzymatic": 1}
    assert select_replacements(network, 100, "enzymatic", 1)[0] == []
    instructions, _ = select_replacements(network, 100, "non_enzymatic", 1)
    assert len(instructions) == 2
    assert ("1.caps.pdb ARG 500 B" in instructions) == (family == "GCP")


def test_cross_family_age_matching_is_rejected(tmp_path):
    caps = {0: tmp_path / "0.caps.pdb", 1: tmp_path / "1.caps.pdb"}
    write_markers(caps[0], [("LGX", 300, "A", {"CE": (0, 0, 0)})])
    write_markers(caps[1], [("LZS", 500, "B", {"NZ1": (1.5, 0, 0)})])
    with pytest.raises(CrosslinkIntegrityError, match="unpaired"):
        resolve_network(caps)


def test_ambiguous_partners_are_not_arbitrarily_paired(tmp_path):
    caps = {i: tmp_path / f"{i}.caps.pdb" for i in range(3)}
    write_markers(caps[0], [("LGX", 300, "A", {"CE": (0, 0, 0)})])
    for i in (1, 2):
        write_markers(caps[i], [("AGS", 500, "B", {"NZ": (i, 0, 0)})])
    with pytest.raises(CrosslinkIntegrityError, match="Ambiguous"):
        resolve_network(caps)
