"""Verify the upstream PYD update alongside the local AGE parametrizations."""

from pathlib import Path

import pytest

from colbuilder.core.topology.crosslink import Crosslink
from colbuilder.core.topology.martini import Martini


def _pdb_atom(serial, atomname, resname, chain, resid, x=0.0, record="ATOM"):
    return (
        f"{record:<6}{serial:5d} {atomname:>4s} {resname:>3s} {chain}{resid:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}\n"
    )


def _force_field_blocks():
    """Read active force-field rows, excluding superseded commented values."""
    path = (
        Path(__file__).parents[1]
        / "src/colbuilder/data/topology/martini300C-ff/aminoacids.ff"
    )
    blocks = {}
    molecule = section = None
    for raw in path.read_text().splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            if section == "moleculetype":
                molecule = None
            elif section in {"link", "modification"}:
                molecule = None
            continue
        fields = line.split()
        if section == "moleculetype":
            molecule = fields[0]
            blocks[molecule] = {}
        elif molecule is not None:
            blocks[molecule].setdefault(section, []).append(fields)
    return blocks


@pytest.mark.parametrize(
    "name, degrees",
    [("al2yx_1", "60"), ("al2yx_2", "100"), ("al2yx_3", "100"), ("al3yx_2", "140")],
)
def test_pyd_uses_upstream_fibril_angle_parameters(name, degrees):
    assert getattr(Crosslink(), name) == degrees


def test_generated_pyd_topology_uses_updated_angles(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lines = []
    for name in ("BB", "SC1", "SC2", "SC3", "SC4", "SC5"):
        lines.append(_pdb_atom(len(lines) + 1, name, "LYX", "A", 10))
    for residue, chain, resid, x in (("LY2", "B", 20, 2.0), ("LY3", "C", 30, 3.0)):
        for name in ("BB", "SC1"):
            lines.append(_pdb_atom(len(lines) + 1, name, residue, chain, resid, x))
    (tmp_path / "0.merge.pdb").write_text("".join(lines) + "END\n")

    bonded = Crosslink(cnt_model=0).set_crosslink_bonded(cnt_model=0)

    assert len(bonded["bonds"]) == 3
    assert {
        tuple(row[:3]): (row[3], row[4], row[5].strip())
        for row in bonded["angles"]
    } == {
        ("6", "5", "8"): ("1", "60", "153"),
        ("4", "5", "8"): ("1", "100", "153"),
        ("5", "8", "7"): ("1", "100", "153"),
        ("5", "6", "10"): ("1", "100", "153"),
        ("4", "6", "10"): ("1", "140", "153"),
        ("6", "10", "9"): ("1", "130", "153"),
    }
    assert len(bonded["angles"]) == 6


def test_lyx_force_field_has_updated_sc2_sc3_sc4_angle():
    rows = [
        row for row in _force_field_blocks()["LYX"]["angles"]
        if row[:3] == ["SC2", "SC3", "SC4"]
    ]
    assert rows == [["SC2", "SC3", "SC4", "2", "160", "150"]]


@pytest.mark.parametrize("terminal_record", ["ATOM", "HETATM"])
def test_chain_lengths_survive_intermediate_and_trailing_ter(terminal_record):
    lines = ["REMARK triple helix\n"]
    for chain, last_residue in (("A", 178), ("B", 181), ("C", 175)):
        lines.extend([
            _pdb_atom(len(lines) + 1, "CA", "GLY", chain, 1),
            _pdb_atom(
                len(lines) + 2, "CA", "GLY", chain, last_residue,
                record=terminal_record,
            ),
            "TER\n",
        ])
    lines.append("END\n")

    assert Martini().get_chain_length(lines) == {"A": " 178", "B": " 181", "C": " 175"}


def test_cap_pdb_renames_only_terminal_ala_with_ter_records():
    lines = []
    for chain, first, last in (("A", "ALA", "ALA"), ("B", "GLY", "ALA"), ("C", "ALA", "GLY")):
        lines.extend([
            _pdb_atom(len(lines) + 1, "CA", first, chain, 1),
            _pdb_atom(len(lines) + 2, "CA", last, chain, 2, record="HETATM"),
            "TER\n",
        ])

    output, cter, nter = Martini().cap_pdb(lines)
    names = {
        (line[21], int(line[22:26])): line[17:20]
        for line in output if line.startswith(("ATOM  ", "HETATM"))
    }
    assert names == {
        ("A", 1): "ALA", ("A", 2): "CLA",
        ("B", 1): "GLY", ("B", 2): "CLA",
        ("C", 1): "ALA", ("C", 2): "GLY",
    }
    assert output.count("TER\n") == 3
    assert (cter, nter) == ("none", "none")


def test_local_glucosepane_g21_fib_r2m_force_field_is_preserved():
    blocks = _force_field_blocks()
    assert blocks["LGX"]["bonds"] == [["BB", "SC1", "1", "0.314", "25000.0"]]
    ags = blocks["AGS"]
    masses = {row[4]: float(row[7]) for row in ags["atoms"] if len(row) > 7}
    assert masses == {"SC2": 66.0, "SC3": 0.0, "SC4": 48.0, "SC5": 36.0, "SC6": 54.0, "SC7": 66.0}
    assert ags["virtual_sites3"] == [["SC3", "SC4", "SC2", "SC7", "4", "0.606", "0.345", "-1.104"]]
    assert ags["dihedrals"] == [["SC5", "SC7", "SC4", "SC2", "2", "-115.1", "82.0"]]
    assert ["SC6", "SC5", "1", "0.337", "12345.0"] in ags["bonds"]
    assert ["SC7", "SC6", "1", "0.285", "23826.0"] in ags["bonds"]
    assert ["SC6", "SC4", "1", "0.353", "25000.0"] in ags["bonds"]

    crosslink = Crosslink()
    assert (crosslink.dgcp_s1_r1, crosslink.kgcp_s1_r1) == ("0.255", "20000.0")
    assert crosslink.agcp_b1_s1_r1 == ("141.879", "127.334")
    assert crosslink.agcp_s1_r1_r3 == ("116.6", "42.0")
    assert crosslink.agcp_s1_r1_r4 == ("132.752", "89.996")


def test_local_mold_force_field_and_cross_marker_parameters_are_preserved():
    blocks = _force_field_blocks()
    assert {row[4]: row[1] for row in blocks["LZS"]["atoms"]} == {
        "BB": "SP2", "SC1": "C1", "SC2": "SN2q", "SC3": "SN4q", "SC4": "TC6"
    }
    assert blocks["LZS"]["bonds"] == [
        ["SC2", "SC3", "1", "0.355", "61818.137"],
        ["SC2", "SC4", "1", "0.401", "56364.559"],
        ["SC2", "SC1", "1", "0.315", "16086.546"],
        ["SC3", "SC4", "1", "0.316", "42615.647"],
        ["SC1", "BB", "1", "0.334", "30653.693"],
    ]
    assert blocks["LZD"]["bonds"] == [["SC1", "BB", "1", "0.334", "28331.059"]]
    crosslink = Crosslink()
    assert (crosslink.dmold_r2_s2, crosslink.kmold_r2_s2) == ("0.310", "18770.020")
    assert crosslink.amold_r1_r2_s2 == ("117.168", "61.307")
    assert crosslink.amold_r3_r2_s2 == ("90.680", "48.532")
    assert crosslink.amold_r2_s2_b2 == ("137.413", "105.966")
