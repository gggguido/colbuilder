import pytest

from colbuilder.core.geometry.crosslink_network import CrosslinkIntegrityError
from colbuilder.core.topology.crosslink_validation import validate_crosslink_terms
from colbuilder.core.topology.amber import Amber
from colbuilder.core.geometry.crosslink import Crosslink
from colbuilder.core.utils.exceptions import TopologyGenerationError


def minimal_pyd():
    return (
        "[ moleculetype ]\ntest 3\n[ atoms ]\n"
        "1 CT 1 LY2 CB 1 0 12\n2 CT 2 LYX C12 2 0 12\n"
        "3 CT 2 LYX C13 3 0 12\n4 CT 3 LY3 CG 4 0 12\n"
        "[ bonds ]\n1 2 1\n2 3 1\n3 4 1\n"
        "[ angles ]\n1 2 3 1\n2 3 4 1\n"
        "[ dihedrals ]\n1 2 3 4 9\n[ pairs ]\n1 4 1\n"
    )


def test_two_pyd_bonds_share_required_1_4_pair(tmp_path):
    path = tmp_path / "pyd.itp"
    path.write_text(minimal_pyd())
    assert validate_crosslink_terms(path, [(1, 2), (3, 4)]) == {
        "bonds": 2,
        "angles": 2,
        "proper_dihedrals": 1,
        "pairs_1_4": 1,
    }


@pytest.mark.parametrize(
    "old,new",
    [
        ("1 2 1\n", ""),
        ("1 2 1\n", "1 2 1\n1 2 1\n"),
        ("1 2 3 1\n", ""),
        ("1 2 3 4 9\n", ""),
        ("1 2 3 4 9\n", "1 2 3 4 4\n"),
        ("1 4 1\n", ""),
        ("1 4 1\n", "1 4 1\n1 4 1\n"),
        ("1 2 1\n", "1 3 1\n"),
    ],
)
def test_missing_duplicate_or_wrong_terms_fail(tmp_path, old, new):
    path = tmp_path / "pyd.itp"
    path.write_text(minimal_pyd().replace(old, new))
    with pytest.raises(CrosslinkIntegrityError):
        validate_crosslink_terms(path, [(1, 2), (3, 4)])


def test_unbonded_markers_cannot_pass_as_valid_single_molecules(tmp_path):
    path = tmp_path / "pyd.itp"
    path.write_text(minimal_pyd().replace("1 2 1\n", "").replace("3 4 1\n", ""))
    with pytest.raises(CrosslinkIntegrityError, match="Unbonded"):
        validate_crosslink_terms(path, [])


def test_wrong_chemical_roles_fail_even_if_expected_ids_are_wrong(tmp_path):
    path = tmp_path / "pyd.itp"
    path.write_text(minimal_pyd().replace("1 2 1\n", "1 3 1\n").replace("3 4 1\n", "2 4 1\n"))
    with pytest.raises(CrosslinkIntegrityError, match="Chemically wrong"):
        validate_crosslink_terms(path, [(1, 3), (2, 4)])


def test_export_refuses_missing_group_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(TopologyGenerationError, match="Missing validated ITP"):
        Amber(ff="amber99sb-star-ildnp").write_topology("out.top", [("T", "0")])
    with pytest.raises(TopologyGenerationError, match="Missing GRO"):
        Amber().write_gro("out.gro", [("T", "0")])


def test_mapping_uses_residue_identity_not_nearest_coordinate(tmp_path):
    path, pdb = tmp_path / "pyd.itp", tmp_path / "input.pdb"
    path.write_text(minimal_pyd())
    amber = Amber()
    amber._merged_residue_maps[str(pdb.resolve())] = (
        {
            (0, "1", "A", "LY2"): 0,
            (1, "2", "B", "LYX"): 1,
            (0, "3", "C", "LY3"): 2,
        },
        3,
    )
    a = Crosslink("2", "LYX", "B", [999, 999, 999], "T", 1, "C13")
    b = Crosslink("3", "LY3", "C", [999, 999, 999], "T", 0, "CG")
    assert amber.find_atom_indices_for_crosslinks(str(path), [(a, b)], str(pdb))[0][1] == (3, 4)


def test_topology_completion_error_is_not_swallowed(tmp_path, monkeypatch):
    path = tmp_path / "pyd.itp"
    path.write_text(minimal_pyd())
    amber = Amber()
    a = Crosslink("2", "LYX", "B", [0, 0, 0], "T", 1, "C12")
    b = Crosslink("1", "LY2", "A", [0, 0, 0], "T", 0, "CB")
    monkeypatch.setattr(amber, "find_atom_indices_for_crosslinks", lambda *args: [((a, b), (2, 1))])
    monkeypatch.setattr(amber, "_add_crosslink_bonds", lambda *args: None)

    def broken(*args):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(amber, "_complete_crosslink_topology", broken)
    with pytest.raises(TopologyGenerationError, match="injected failure"):
        amber.add_crosslink_topology_to_itp(str(path), [(a, b)], "input.pdb")


def test_crosslink_mapping_preserves_insertion_code_identity(tmp_path):
    path, pdb = tmp_path / "pyd.itp", tmp_path / "input.pdb"
    path.write_text(minimal_pyd())
    amber = Amber()
    amber._merged_residue_maps[str(pdb.resolve())] = (
        {(0, "1", "A", "LY2"): 0, (1, "2A", "B", "LYX"): 1, (0, "3", "C", "LY3"): 2},
        3,
        ["1", "2", "3"],
    )
    a = Crosslink("2A", "LYX", "B", [0, 0, 0], "T", 1, "C13")
    b = Crosslink("3", "LY3", "C", [0, 0, 0], "T", 0, "CG")
    assert amber.find_atom_indices_for_crosslinks(str(path), [(a, b)], str(pdb))[0][1] == (3, 4)
