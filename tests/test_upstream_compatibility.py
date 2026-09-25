"""Regressions for upstream topology fixes combined with the local safeguards."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from colbuilder.core.topology.amber import Amber
from colbuilder.core.topology.martini import Martini


class _System:
    def __init__(self, connections=(1, 2)):
        self.model = SimpleNamespace(type="D", connect=connections)

    def get_model(self, model_id):
        return self.model


def _pdb_atom(serial, atom="CA", chain="A", residue=1):
    return (
        f"ATOM  {serial:5d} {atom:>4} GLY {chain}{residue:4d}    "
        f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{0.0:6.2f}           C\n"
    )


def test_martini_rejects_incomplete_connected_group(tmp_path, monkeypatch):
    """One surviving partner must not produce a seemingly complete topology."""
    monkeypatch.chdir(tmp_path)
    surviving_input = tmp_path / "7.1.CG.pdb"
    surviving_input.write_text(_pdb_atom(1) + "TER\nEND\n")

    result = Martini(system=_System()).merge_pdbs(model_id=7, cnt_model=4)

    assert result is None
    assert not (tmp_path / "4.merge.pdb").exists()
    assert surviving_input.is_file()


@pytest.mark.parametrize("connections", [(1, 2), ()])
def test_martini_preserves_unpadded_ter_in_read_and_merge(
    tmp_path, monkeypatch, connections
):
    """Keep chain boundaries both with connectivity and the disk fallback."""
    monkeypatch.chdir(tmp_path)
    topology = Martini(system=_System(connections))
    source = _pdb_atom(1) + "TER\n" + _pdb_atom(2, chain="B") + "TER\n"
    caps_dir = tmp_path / "D"
    caps_dir.mkdir()
    (caps_dir / "1.caps.pdb").write_text("REMARK source\n" + source + "END\n")

    assert topology.read_pdb(pdb_id=1) == source.splitlines(keepends=True)

    for connect_id in (1, 2):
        (tmp_path / f"7.{connect_id}.CG.pdb").write_text(source + "END\n")
    result = topology.merge_pdbs(model_id=7, cnt_model=4)

    assert result is not None
    assert Path(result).read_text() == source + source + "END\n"


def test_amber_preserves_unpadded_ter_while_merging_caps(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    caps_dir = tmp_path / "D"
    caps_dir.mkdir()
    source = _pdb_atom(1) + "TER\n" + _pdb_atom(2, chain="B") + "TER\n"
    (caps_dir / "1.caps.pdb").write_text(source + "END\n")

    result = Amber(system=_System((1,))).merge_connected_models([1])

    assert result is not None
    assert (caps_dir / "1.merge.pdb").read_text() == source + "END\n"


def test_amber_gro_box_contains_all_groups_with_padding(tmp_path, monkeypatch):
    """Separated helices need the full coordinate extent, not the last box."""
    monkeypatch.chdir(tmp_path)
    atom_lines = []
    for group_id, xyz in (("0", (-3.0, -1.0, 2.0)), ("1", (9.0, 5.0, 20.0))):
        atom_line = (
            f"{1:5d}{'GLY':<5}{'CA':>5}{1:5d}"
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}\n"
        )
        atom_lines.append(atom_line)
        (tmp_path / f"col_{group_id}.gro").write_text(
            "group\n1\n" + atom_line + "   1.00000   1.00000   1.00000\n"
        )
        # The local code also checks GRO/ITP identity and ordering before export.
        (tmp_path / f"col_{group_id}.itp").write_text(
            "[ moleculetype ]\ncol 3\n[ atoms ]\n1 CT 1 GLY CA 1 0 12\n"
        )

    Amber().write_gro("merged.gro", [("D", "0"), ("D", "1")])

    lines = (tmp_path / "merged.gro").read_text().splitlines(keepends=True)
    assert int(lines[1]) == 2
    assert lines[2:-1] == atom_lines
    assert [float(value) for value in lines[-1].split()] == pytest.approx(
        [14.0, 8.0, 20.0]
    )


@pytest.mark.parametrize(
    "oxt_residue, expected_cap", [(None, "NME"), (1, "NME"), (2, "none")]
)
def test_martini_does_not_add_nme_to_oxt_terminated_residue(
    oxt_residue, expected_cap
):
    """OXT must suppress NME only when it belongs to a terminal residue."""
    pdb = []
    for chain in "ABC":
        for residue in (1, 2):
            pdb.append(_pdb_atom(len(pdb) + 1, chain=chain, residue=residue))
            if chain == "A" and residue == oxt_residue:
                pdb.append(
                    _pdb_atom(len(pdb) + 1, atom="OXT", chain=chain, residue=residue)
                )
        pdb.append("TER\n")

    output, cter, nter = Martini().cap_pdb(pdb.copy())

    assert output == pdb
    assert cter == expected_cap
    assert nter == "none"
