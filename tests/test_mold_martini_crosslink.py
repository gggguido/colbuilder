from pathlib import Path

import pytest

from colbuilder.core.topology.crosslink import Crosslink, IncompleteCrosslinkError
from colbuilder.core.topology.itp import Itp


def _pdb_atom(serial, atomname, resname, chain, resid, x, y, z):
    return (
        f"ATOM  {serial:5d} {atomname:>4s} {resname:>3s} {chain}{resid:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}\n"
    )


def test_mold_cross_marker_terms_use_the_paired_marker_beads(tmp_path, monkeypatch):
    """MOLD's inter-marker bond and three angles are inserted exactly once."""
    lines = [
        _pdb_atom(1, "BB", "LZS", "A", 10, -1.0, 0.0, 0.0),
        _pdb_atom(2, "SC1", "LZS", "A", 10, -0.5, 0.0, 0.0),
        _pdb_atom(3, "SC2", "LZS", "A", 10, -0.2, 0.0, 0.0),
        _pdb_atom(4, "SC3", "LZS", "A", 10, 0.0, 0.0, 0.0),
        _pdb_atom(5, "SC4", "LZS", "A", 10, 0.0, 0.3, 0.0),
        _pdb_atom(6, "BB", "LZD", "B", 20, 1.0, 0.0, 0.0),
        _pdb_atom(7, "SC1", "LZD", "B", 20, 0.3, 0.0, 0.0),
        "END\n",
    ]
    pdb_path = Path(tmp_path) / "0.merge.pdb"
    pdb_path.write_text("".join(lines))
    monkeypatch.chdir(tmp_path)

    bonded = Crosslink(cnt_model=0).set_crosslink_bonded(cnt_model=0)

    assert bonded["bonds"] == [["4", "7", "1", "0.310", "18770.020\n"]]
    assert bonded["angles"] == [
        ["3", "4", "7", "1", "117.168", "61.307\n"],
        ["5", "4", "7", "1", "90.680", "48.532\n"],
        ["4", "7", "6", "1", "137.413", "105.966\n"],
    ]
    assert bonded["dihedrals"] == []


def test_mold_orphan_marker_is_rejected(tmp_path, monkeypatch):
    """A MOLD marker half must not silently become a disconnected side chain."""
    pdb_path = Path(tmp_path) / "0.merge.pdb"
    pdb_path.write_text(
        _pdb_atom(1, "BB", "LZS", "A", 10, -1.0, 0.0, 0.0)
        + _pdb_atom(2, "SC3", "LZS", "A", 10, 0.0, 0.0, 0.0)
        + "END\n"
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(IncompleteCrosslinkError, match="Incomplete MOLD mapping"):
        Crosslink(cnt_model=0).set_crosslink_bonded(cnt_model=0)


def test_mold_missing_angle_bead_is_rejected(tmp_path, monkeypatch):
    """Every matched pair must produce the complete one-bond/three-angle bridge."""
    lines = [
        _pdb_atom(1, "BB", "LZS", "A", 10, -1.0, 0.0, 0.0),
        # LZS:SC2/R1 is deliberately absent.
        _pdb_atom(2, "SC3", "LZS", "A", 10, 0.0, 0.0, 0.0),
        _pdb_atom(3, "SC4", "LZS", "A", 10, 0.0, 0.3, 0.0),
        _pdb_atom(4, "BB", "LZD", "B", 20, 1.0, 0.0, 0.0),
        _pdb_atom(5, "SC1", "LZD", "B", 20, 0.3, 0.0, 0.0),
        "END\n",
    ]
    pdb_path = Path(tmp_path) / "0.merge.pdb"
    pdb_path.write_text("".join(lines))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(
        IncompleteCrosslinkError, match="Incomplete MOLD bonded topology"
    ):
        Crosslink(cnt_model=0).set_crosslink_bonded(cnt_model=0)


def test_incomplete_mold_aborts_itp_generation(tmp_path, monkeypatch):
    """The ITP merge layer must propagate MOLD completeness errors."""
    pdb_path = Path(tmp_path) / "0.merge.pdb"
    pdb_path.write_text(
        _pdb_atom(1, "BB", "LZS", "A", 10, -1.0, 0.0, 0.0)
        + _pdb_atom(2, "SC3", "LZS", "A", 10, 0.0, 0.0, 0.0)
        + "END\n"
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(IncompleteCrosslinkError, match="Incomplete MOLD mapping"):
        Itp().make_topology(model_id=0, cnt_model=0)

    assert not (tmp_path / "col_0.itp").exists()


def test_mold_mapping_hosts_only_the_available_part_of_shared_r2():
    """The residue-local maps implement the documented inter-marker split."""
    mapping_dir = (
        Path(__file__).parents[1]
        / "src/colbuilder/data/topology/martini300C-mapping"
    )
    lzs = (mapping_dir / "lzs.amber99.map").read_text()
    lzd = (mapping_dir / "lzd.amber99.map").read_text()

    assert "NZ1    SC3" in lzs
    assert "C31    SC3" in lzs
    assert "CE   !SC1" in lzd
    assert "HE1   !SC1" in lzd
    assert "HE2   !SC1" in lzd
