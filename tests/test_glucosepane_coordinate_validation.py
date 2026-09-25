import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from colbuilder.core.sequence.sequence_generator import SequenceGenerator
from colbuilder.core.topology import coordinate_validation
from colbuilder.core.topology.coordinate_validation import validate_glucosepane_contacts
from colbuilder.core.utils.exceptions import (
    SequenceGenerationError,
    TopologyGenerationError,
)


@pytest.fixture(autouse=True)
def capture_contact_warnings(monkeypatch):
    monkeypatch.setattr(coordinate_validation, "LOG", logging.getLogger("test_gcp_contacts"))


def write_system(tmp_path, distance, hydrogen=False, bonded=False, split=False):
    atoms = [("AGS", "NE"), ("GLU", "HB1" if hydrogen else "CB")]
    paths = []
    for index, chunk in enumerate([[a] for a in atoms] if split else [atoms]):
        path = tmp_path / f"col_{index}.itp"
        text = "[ moleculetype ]\ntest 3\n[ atoms ]\n"
        for i, (residue, name) in enumerate(chunk, 1):
            text += f"{i} CT {i} {residue} {name} 1 0 12\n"
        if bonded:
            text += "[ bonds ]\n1 2 1\n"
        path.write_text(text)
        paths.append(path)
    gro = tmp_path / "test.gro"
    lines = ["test\n", "2\n"]
    for i, (residue, name) in enumerate(atoms, 1):
        x = 0 if i == 1 else distance / 10
        lines.append(f"{i:5d}{residue:<5}{name:>5}{i:5d}{x:8.3f}{0:8.3f}{0:8.3f}\n")
    gro.write_text("".join(lines) + "10 10 10\n")
    return gro, paths


@pytest.mark.parametrize("split", [False, True])
def test_extreme_hydrogen_overlap_warns_across_itps(tmp_path, split, caplog):
    gro, paths = write_system(tmp_path, 0.124, hydrogen=True, split=split)
    original = {path: path.read_bytes() for path in [gro, *paths]}
    result = validate_glucosepane_contacts(gro, paths)
    assert result["gross_overlaps"] == 1
    assert result["compressed_contacts"] == 0
    assert "Export continues" in caplog.text
    assert "0.1200 A" in caplog.text  # GRO precision.
    assert "global atom 1 AGS1:NE (type CT, col_0.itp, local atom 1)" in caplog.text
    second = "GLU1:HB1 (type CT, col_1.itp, local atom 1)" if split else (
        "GLU2:HB1 (type CT, col_0.itp, local atom 2)"
    )
    assert f"global atom 2 {second}" in caplog.text
    assert all(record.levelno == logging.WARNING for record in caplog.records)
    assert all(path.read_bytes() == content for path, content in original.items())


def test_heavy_overlap_warns(tmp_path, caplog):
    gro, paths = write_system(tmp_path, 0.2)
    assert validate_glucosepane_contacts(gro, paths)["gross_overlaps"] == 1
    assert "0.2000 A" in caplog.text
    assert "GLU2:CB" in caplog.text


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("distance,hydrogen", [
    (0.25, False), (0.323, False), (0.65, False), (1.49, False), (1.99, False),
    (0.15, True), (0.4, True), (0.59, True),
])
def test_compressed_contacts_warn_without_blocking_export(tmp_path, distance, hydrogen, split, caplog):
    gro, paths = write_system(tmp_path, distance, hydrogen=hydrogen, split=split)
    result = validate_glucosepane_contacts(gro, paths)
    assert result["gross_overlaps"] == 0
    assert result["compressed_contacts"] == 1
    assert "compressed nonbonded contacts" in caplog.text
    assert "global atom 1 AGS1:NE" in caplog.text
    assert "Export continues" in caplog.text


@pytest.mark.parametrize("distance,hydrogen", [(0.0, False), (0.0, True),
                                              (0.24, False), (0.14, True)])
def test_severe_contacts_remain_counted_without_blocking_export(tmp_path, distance, hydrogen):
    gro, paths = write_system(tmp_path, distance, hydrogen=hydrogen)
    result = validate_glucosepane_contacts(gro, paths)
    assert result["gross_overlaps"] == 1
    assert result["compressed_contacts"] == 0


def test_all_severe_contacts_are_identified_once(tmp_path, caplog):
    gro, paths = write_system(tmp_path, 0.2)
    paths[0].write_text(paths[0].read_text() + "3 OH 494 AGS O18 1 0 16\n")
    rows = gro.read_text().splitlines(True)
    rows[1] = "3\n"
    rows.insert(-1, f"{494:5d}{'AGS':<5}{'O18':>5}{3:5d}{0.01:8.3f}{0:8.3f}{0:8.3f}\n")
    gro.write_text("".join(rows))
    result = validate_glucosepane_contacts(gro, paths)
    assert result["gross_overlaps"] == 3
    details = [r.getMessage() for r in caplog.records if "overlap warning:" in r.getMessage()]
    assert len(details) == 3
    assert any("AGS494:O18 (type OH" in message for message in details)


@pytest.mark.parametrize("graph_distance", [1, 2, 3])
def test_covalent_exemptions_and_severe_14_warning(tmp_path, graph_distance, caplog):
    itp, gro = tmp_path / 'col_0.itp', tmp_path / 'test.gro'
    names = [('AGS', 'NE')] + [('GLY', 'C')] * graph_distance
    rows = []
    atom_lines = []
    for i, (residue, name) in enumerate(names, 1):
        atom_lines.append(f'{i} CT {i} {residue} {name} {i} 0 12\n')
        x = 0 if i == 1 else 0.02 if i == len(names) else float(i)
        rows.append(f'{i:5d}{residue:<5}{name:>5}{i:5d}{x:8.3f}{0:8.3f}{0:8.3f}\n')
    bonds = ''.join(f'{i} {i + 1} 1\n' for i in range(1, len(names)))
    itp.write_text('[ atoms ]\n' + ''.join(atom_lines) + '[ bonds ]\n' + bonds)
    gro.write_text('test\n' + str(len(names)) + '\n' + ''.join(rows) + '10 10 10\n')
    result = validate_glucosepane_contacts(gro, [itp])
    assert result['gross_overlaps'] == int(graph_distance == 3)
    assert result['compressed_contacts'] == 0
    assert bool(caplog.records) == (graph_distance == 3)


def test_bonds_are_not_treated_as_nonbonded_contacts(tmp_path):
    gro, paths = write_system(tmp_path, 1.4, bonded=True)
    assert validate_glucosepane_contacts(gro, paths)["gross_overlaps"] == 0


def test_separated_atoms_pass(tmp_path):
    gro, paths = write_system(tmp_path, 3.0)
    assert validate_glucosepane_contacts(gro, paths)["gross_overlaps"] == 0


def test_atom_mapping_mismatch_fails(tmp_path):
    gro, paths = write_system(tmp_path, 3.0)
    gro.write_text(gro.read_text().replace("AGS", "LYS"))
    with pytest.raises(TopologyGenerationError, match="atom order mismatch"):
        validate_glucosepane_contacts(gro, paths)


def test_atom_count_mismatch_still_fails(tmp_path):
    gro, paths = write_system(tmp_path, 3.0)
    gro.write_text(gro.read_text().replace('test\n2\n', 'test\n3\n'))
    with pytest.raises(TopologyGenerationError, match='atom count mismatch'):
        validate_glucosepane_contacts(gro, paths)


def test_noncontiguous_itp_atoms_still_fail(tmp_path):
    gro, paths = write_system(tmp_path, 3.0)
    paths[0].write_text(paths[0].read_text().replace('2 CT', '3 CT'))
    with pytest.raises(TopologyGenerationError, match='Noncontiguous atom numbering'):
        validate_glucosepane_contacts(gro, paths)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_coordinates_still_fail(tmp_path, value):
    gro, paths = write_system(tmp_path, value)
    with pytest.raises(TopologyGenerationError, match='Nonfinite coordinates'):
        validate_glucosepane_contacts(gro, paths)


def test_steric_error_is_not_misreported_as_directory_setup_error(tmp_path):
    generator = SequenceGenerator.__new__(SequenceGenerator)
    generator.config = SimpleNamespace(working_directory=tmp_path, debug=True)
    generator.file_manager = SimpleNamespace(get_temp_path=lambda *a, **k: tmp_path)
    original = Path.cwd()
    error = SequenceGenerationError(
        "No sterically acceptable crosslink geometry found", error_code="SEQ_ERR_003"
    )

    async def run():
        async with generator._manage_resources():
            raise error

    with pytest.raises(SequenceGenerationError) as caught:
        asyncio.run(run())
    assert caught.value is error
    assert Path.cwd() == original
