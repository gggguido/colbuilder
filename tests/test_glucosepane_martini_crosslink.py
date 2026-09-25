from pathlib import Path

import pytest

from colbuilder.core.topology.crosslink import Crosslink, IncompleteCrosslinkError
from colbuilder.core.topology.itp import Itp


def _pdb_atom(serial, atomname, resname, chain, resid, x, y, z):
    return (
        f"ATOM  {serial:5d} {atomname:>4s} {resname:>3s} {chain}{resid:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}\n"
    )


def test_glucosepane_cross_marker_terms_use_the_paired_marker_beads(
    tmp_path, monkeypatch
):
    """The only inter-marker bond and the three cross-boundary angles of the
    glucosepane model (G21-fib R2M) are inserted exactly once."""
    lines = [
        _pdb_atom(1, "BB", "LGX", "A", 10, -1.0, 0.0, 0.0),
        _pdb_atom(2, "SC1", "LGX", "A", 10, 0.0, 0.0, 0.0),
        _pdb_atom(3, "BB", "AGS", "B", 20, 1.0, 1.0, 0.0),
        _pdb_atom(4, "SC1", "AGS", "B", 20, 1.0, 0.5, 0.0),
        _pdb_atom(5, "SC2", "AGS", "B", 20, 0.8, 0.4, 0.0),
        _pdb_atom(6, "SC3", "AGS", "B", 20, 0.6, 0.3, 0.0),
        _pdb_atom(7, "SC4", "AGS", "B", 20, 0.4, 0.2, 0.0),
        _pdb_atom(8, "SC5", "AGS", "B", 20, 0.2, 0.0, 0.0),
        _pdb_atom(9, "SC6", "AGS", "B", 20, 0.3, -0.2, 0.0),
        _pdb_atom(10, "SC7", "AGS", "B", 20, 0.5, -0.3, 0.0),
        "END\n",
    ]
    pdb_path = Path(tmp_path) / "0.merge.pdb"
    pdb_path.write_text("".join(lines))
    monkeypatch.chdir(tmp_path)

    bonded = Crosslink(cnt_model=0).set_crosslink_bonded(cnt_model=0)

    # G21 (superseded): bond 0.295/63395.824 and angle S1--R1--R2 to SC6
    # (["2", "8", "9", "1", "89.976", "48.526\n"]).
    assert bonded["bonds"] == [["2", "8", "1", "0.255", "20000.0\n"]]
    assert bonded["angles"] == [
        ["1", "2", "8", "1", "141.879", "127.334\n"],
        ["2", "8", "10", "1", "116.6", "42.0\n"],
        ["2", "8", "7", "1", "132.752", "89.996\n"],
    ]
    assert bonded["dihedrals"] == []


def test_glucosepane_orphan_marker_is_rejected(tmp_path, monkeypatch):
    """A marker half must not silently become a disconnected side chain."""
    pdb_path = Path(tmp_path) / "0.merge.pdb"
    pdb_path.write_text(
        _pdb_atom(1, "BB", "LGX", "A", 10, -1.0, 0.0, 0.0)
        + _pdb_atom(2, "SC1", "LGX", "A", 10, 0.0, 0.0, 0.0)
        + "END\n"
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(IncompleteCrosslinkError, match="Incomplete glucosepane mapping"):
        Crosslink(cnt_model=0).set_crosslink_bonded(cnt_model=0)


def test_glucosepane_missing_constructor_bead_is_rejected(tmp_path, monkeypatch):
    """Every matched pair must produce the full one-bond/three-angle bridge."""
    lines = [
        _pdb_atom(1, "BB", "LGX", "A", 10, -1.0, 0.0, 0.0),
        _pdb_atom(2, "SC1", "LGX", "A", 10, 0.0, 0.0, 0.0),
        _pdb_atom(3, "BB", "AGS", "B", 20, 1.0, 1.0, 0.0),
        _pdb_atom(4, "SC4", "AGS", "B", 20, 0.4, 0.2, 0.0),
        _pdb_atom(5, "SC5", "AGS", "B", 20, 0.2, 0.0, 0.0),
        # SC7/R3 (and SC6/R2) are deliberately absent.
        "END\n",
    ]
    pdb_path = Path(tmp_path) / "0.merge.pdb"
    pdb_path.write_text("".join(lines))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(
        IncompleteCrosslinkError, match="Incomplete glucosepane bonded topology"
    ):
        Crosslink(cnt_model=0).set_crosslink_bonded(cnt_model=0)


def test_incomplete_glucosepane_aborts_itp_generation(tmp_path, monkeypatch):
    """The ITP merge layer must not swallow a glucosepane completeness error."""
    pdb_path = Path(tmp_path) / "0.merge.pdb"
    pdb_path.write_text(
        _pdb_atom(1, "BB", "LGX", "A", 10, -1.0, 0.0, 0.0)
        + _pdb_atom(2, "SC1", "LGX", "A", 10, 0.0, 0.0, 0.0)
        + "END\n"
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(IncompleteCrosslinkError, match="Incomplete glucosepane mapping"):
        Itp().make_topology(model_id=0, cnt_model=0)

    assert not (tmp_path / "col_0.itp").exists()
