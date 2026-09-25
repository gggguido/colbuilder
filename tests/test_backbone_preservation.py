"""Regression tests for TER-induced chain breaks during crosslink replacement."""

import pytest

from colbuilder.core.geometry.backbone import (
    BackboneIntegrityError,
    preserve_backbone,
    read_pdb_residues,
    validate_backbone_topology,
)


def atom(serial, name, residue, number, chain="C", xyz=(0.0, 0.0, 0.0), insertion=""):
    element = name.lstrip("0123456789")[0]
    return (
        f"ATOM  {serial:5d} {name:^4s} {residue:>3s} {chain}{number:4d}{insertion:1s}   "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {element:>2s}\n"
    )


def pdb(name="LYX", new_ter=False, true_ter=False, next_chain="C", insertion=""):
    names = ["N", "CA", "C", "O"] + (
        ["CB", "CG", "CD", "CE", "NZ"] if name == "LYS" else ["CB", "C12", "C13"]
    )
    rows = [
        atom(i, a, name, 103, xyz=(0.0, 0.0, 0.0), insertion=insertion)
        for i, a in enumerate(names, 1)
    ]
    if new_ter or true_ter:
        rows.append("TER\n")
    # Deliberately long C-N, matching the failure mechanism; never use distance
    # to decide whether these consecutive residues are covalently connected.
    rows.extend(
        atom(i + len(names) + 1, a, "GLY", 104, chain=next_chain, xyz=(2.30, 0.0, 0.0))
        for i, a in enumerate(["N", "CA", "C", "O"])
    )
    rows += ["TER\n", "END\n"]
    return "".join(rows)


def residues(path):
    return read_pdb_residues(path.read_text().splitlines(keepends=True), str(path))


def test_long_peptide_bond_does_not_become_chain_break(tmp_path):
    path = tmp_path / "48.caps.pdb"
    original = pdb()
    path.write_text(original)
    with preserve_backbone(tmp_path, ["48.caps.pdb LYS 103 C"]):
        path.write_text(pdb("LYS", new_ter=True))
    first, second = residues(path)
    assert first.segment == second.segment
    assert first.name == "LYS"
    assert first.atoms["C"][30:54] == residues_from_text(original)[0].atoms["C"][30:54]
    assert second.atoms["N"][30:54] == residues_from_text(original)[1].atoms["N"][30:54]
    assert "C12" not in first.atoms
    assert set(first.atoms) == {"N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"}


def residues_from_text(text):
    return read_pdb_residues(text.splitlines(keepends=True), "test")


@pytest.mark.parametrize("true_ter,next_chain", [(True, "C"), (False, "B"), (True, "B")])
def test_real_boundaries_survive_even_if_chimera_omits_ter(tmp_path, true_ter, next_chain):
    path = tmp_path / "48.caps.pdb"
    path.write_text(pdb(true_ter=true_ter, next_chain=next_chain))
    with preserve_backbone(tmp_path, ["48.caps.pdb LYS 103 C"]):
        path.write_text(pdb("LYS", next_chain=next_chain))
    first, second = residues(path)
    assert first.segment != second.segment


def test_missing_backbone_O_is_restored(tmp_path):
    path = tmp_path / "48.caps.pdb"
    path.write_text(pdb())
    with preserve_backbone(tmp_path, ["48.caps.pdb LYS 103 C"]):
        path.write_text(
            "".join(
                line
                for line in pdb("LYS", new_ter=True).splitlines(keepends=True)
                if not (
                    line.startswith("ATOM") and line[17:20] == "LYS" and line[12:16].strip() == "O"
                )
            )
        )
    assert "O" in residues(path)[0].atoms


def test_unrequested_coordinates_are_never_taken_from_chimera(tmp_path):
    path = tmp_path / "48.caps.pdb"
    path.write_text(pdb())
    original_gly = residues(path)[1]
    with preserve_backbone(tmp_path, ["48.caps.pdb LYS 103 C"]):
        path.write_text(pdb("LYS").replace("   2.300", "  12.300"))
    assert {k: v[30:54] for k, v in residues(path)[1].atoms.items()} == {
        k: v[30:54] for k, v in original_gly.atoms.items()
    }


def test_ARG_sidechain_and_multiple_mutations_in_same_file(tmp_path):
    path = tmp_path / "48.caps.pdb"
    path.write_text(pdb("AGS"))
    arg_names = ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"]
    lys_names = ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"]
    changed = [atom(i, name, "ARG", 103) for i, name in enumerate(arg_names, 1)]
    changed.append("TER\n")
    changed += [
        atom(i + 12, name, "LYS", 104, xyz=(2.3, 0.0, 0.0)) for i, name in enumerate(lys_names)
    ]
    with preserve_backbone(tmp_path, ["48.caps.pdb ARG 103 C", "48.caps.pdb LYS 104 C"]):
        path.write_text("".join(changed) + "TER\nEND\n")
    a, b = residues(path)
    assert a.name == "ARG" and b.name == "LYS"
    assert a.segment == b.segment
    assert set(a.atoms) == set(arg_names)
    assert set(b.atoms) == set(lys_names)


def test_caps_are_preserved_and_not_rebuilt_by_chimera(tmp_path):
    path = tmp_path / "48.caps.pdb"
    ace = atom(40, "C", "ACE", 102) + atom(41, "O", "ACE", 102)
    nme = atom(42, "N", "NME", 105) + atom(43, "CH3", "NME", 105)
    source = ace + pdb().replace("TER\nEND\n", nme + "TER\nEND\n")
    changed = ace + pdb("LYS", new_ter=True).replace("TER\nEND\n", nme + "TER\nEND\n")
    path.write_text(source)
    with preserve_backbone(tmp_path, ["48.caps.pdb LYS 103 C"]):
        path.write_text(changed)
    before, after = residues_from_text(source), residues(path)
    assert [r.name for r in after] == ["ACE", "LYS", "GLY", "NME"]
    assert len({r.segment for r in after}) == 1
    for i in (0, 3):
        assert {k: v[11:] for k, v in before[i].atoms.items()} == {
            k: v[11:] for k, v in after[i].atoms.items()
        }


@pytest.mark.parametrize("ambiguity", ["alternate", "repeated", "models", "missing_backbone"])
def test_ambiguous_inputs_are_rejected_without_running_mutation(tmp_path, ambiguity):
    path = tmp_path / "48.caps.pdb"
    original = pdb()
    if ambiguity == "alternate":
        original = original[:16] + "A" + original[17:]
    elif ambiguity == "repeated":
        original += pdb()
    elif ambiguity == "models":
        original = (
            "MODEL        1\n" + original + "ENDMDL\nMODEL        2\n" + original + "ENDMDL\n"
        )
    else:
        original = "".join(
            line
            for line in original.splitlines(keepends=True)
            if not (line[17:20] == "LYX" and line[12:16].strip() == "O")
        )
    path.write_text(original)
    with pytest.raises(BackboneIntegrityError):
        with preserve_backbone(tmp_path, ["48.caps.pdb LYS 103 C"]):
            pytest.fail("invalid input reached Chimera")
    assert path.read_text() == original


def test_preexisting_numbering_gap_and_TER_are_preserved(tmp_path):
    path = tmp_path / "48.caps.pdb"
    path.write_text(pdb(true_ter=True).replace("GLY C 104", "GLY C 110"))
    with preserve_backbone(tmp_path, ["48.caps.pdb LYS 103 C"]):
        path.write_text(pdb("LYS").replace("GLY C 104", "GLY C 110"))
    a, b = residues(path)
    assert b.address[2] == "110" and a.segment != b.segment


@pytest.mark.parametrize(
    "failure",
    ["wrong_residue", "missing_sidechain", "missing_residue", "reordered", "moved_backbone"],
)
def test_invalid_mutation_rolls_back_the_whole_batch(tmp_path, failure):
    original = pdb()
    paths = [tmp_path / name for name in ("1.caps.pdb", "2.caps.pdb")]
    for path in paths:
        path.write_text(original)
    bad = pdb("LYS", new_ter=True)
    if failure == "wrong_residue":
        bad = pdb()
    elif failure == "missing_sidechain":
        bad = "".join(line for line in bad.splitlines(keepends=True) if line[12:16].strip() != "NZ")
    elif failure == "missing_residue":
        bad = "".join(line for line in bad.splitlines(keepends=True) if line[17:20] != "GLY")
    elif failure == "reordered":
        lines = bad.splitlines(keepends=True)
        bad = "".join(
            [line for line in lines if line[17:20] == "GLY"]
            + [line for line in lines if line[17:20] == "LYS"]
            + ["END\n"]
        )
    else:
        bad = bad.replace("   0.000", "   5.000", 1)
    with pytest.raises(BackboneIntegrityError):
        with preserve_backbone(tmp_path, [f"{p.name} LYS 103 C" for p in paths]):
            paths[0].write_text(pdb("LYS"))
            paths[1].write_text(bad)
    assert all(p.read_text() == original for p in paths)


def test_subprocess_failure_restores_deleted_files(tmp_path):
    path = tmp_path / "48.caps.pdb"
    original = pdb()
    path.write_text(original)
    with pytest.raises(RuntimeError, match="Chimera failed"):
        with preserve_backbone(tmp_path, ["48.caps.pdb LYS 103 C"]):
            path.unlink()
            raise RuntimeError("Chimera failed")
    assert path.read_text() == original


def test_insertion_code_and_case_insensitive_chain(tmp_path):
    path = tmp_path / "48.caps.pdb"
    path.write_text(pdb(insertion="A"))
    with preserve_backbone(tmp_path, ["48.caps.pdb LYS 103A c"]):
        path.write_text(pdb("LYS", new_ter=True, insertion="A"))
    assert residues(path)[0].address[1:] == ("C", "103", "A")


def test_serials_and_conect_do_not_point_to_removed_marker_atoms(tmp_path):
    path = tmp_path / "48.caps.pdb"
    path.write_text(pdb().replace("END\n", "CONECT    3    8\nCONECT    5    6\nEND\n"))
    with preserve_backbone(tmp_path, ["48.caps.pdb LYS 103 C"]):
        path.write_text(pdb("LYS", new_ter=True))
    rr = residues(path)
    c = int(rr[0].atoms["C"][6:11])
    n = int(rr[1].atoms["N"][6:11])
    assert [line for line in path.read_text().splitlines() if line.startswith("CONECT")] == [
        f"CONECT{c:5d}{n:5d}"
    ]
    ids = [
        int(line[6:11])
        for line in path.read_text().splitlines()
        if line.startswith(("ATOM", "TER"))
    ]
    assert ids == list(range(1, len(ids) + 1))


@pytest.mark.parametrize(
    "instruction",
    ["48.caps.pdb LYS 999 C", "48.caps.pdb ALA 103 C", "invalid", "../other.pdb LYS 103 C"],
)
def test_invalid_targets_fail_before_mutation(tmp_path, instruction):
    path = tmp_path / "48.caps.pdb"
    path.write_text(pdb())
    with pytest.raises(BackboneIntegrityError):
        with preserve_backbone(tmp_path, [instruction]):
            pytest.fail("mutation must not start")


def write_itp(path, peptide=True, terminal=False):
    rows = ["[ moleculetype ]\ncol 3\n[ atoms ]\n"]
    for i, (res, number) in enumerate([("LYS", 103), ("GLY", 104)]):
        for j, atom_name in enumerate(["N", "CA", "C", "OC1" if terminal and i == 0 else "O"], 1):
            n = i * 4 + j
            rows.append(f"{n} CT {number} {res} {atom_name} {n} 0 12\n")
    rows.append("[ bonds ]\n1 2 1\n2 3 1\n3 4 1\n5 6 1\n6 7 1\n7 8 1\n")
    if peptide:
        rows.append("3 5 1\n")
    path.write_text("".join(rows))


def test_validator_accepts_long_but_connected_backbone(tmp_path):
    p, t = tmp_path / "input.pdb", tmp_path / "col.itp"
    p.write_text(pdb("LYS"))
    write_itp(t)
    validate_backbone_topology(p, t)


def test_validator_rejects_missing_peptide_even_when_grompp_would_accept(tmp_path):
    p, t = tmp_path / "input.pdb", tmp_path / "col.itp"
    p.write_text(pdb("LYS"))
    write_itp(t, peptide=False)
    with pytest.raises(BackboneIntegrityError, match="missing peptide bond"):
        validate_backbone_topology(p, t)


def test_validator_rejects_false_terminal_inside_polymer(tmp_path):
    p, t = tmp_path / "input.pdb", tmp_path / "col.itp"
    p.write_text(pdb("LYS"))
    write_itp(t, peptide=False, terminal=True)
    with pytest.raises(BackboneIntegrityError, match="missing peptide bond"):
        validate_backbone_topology(p, t)


def test_validator_accepts_true_termini_but_rejects_joining_them(tmp_path):
    p, t = tmp_path / "input.pdb", tmp_path / "col.itp"
    p.write_text(pdb("LYS", true_ter=True))
    write_itp(t, peptide=False, terminal=True)
    validate_backbone_topology(p, t)
    write_itp(t, peptide=True, terminal=True)
    with pytest.raises(BackboneIntegrityError, match="across input polymer boundaries"):
        validate_backbone_topology(p, t)


def test_amber_aborts_instead_of_exporting_a_broken_group(tmp_path):
    from colbuilder.core.topology.amber import Amber
    from colbuilder.core.utils.exceptions import TopologyGenerationError

    p, top = tmp_path / "input.pdb", tmp_path / "col.top"
    p.write_text(pdb("LYS"))
    write_itp(top, peptide=False)
    top.write_text(top.read_text().replace("col 3", "Protein 3"))
    with pytest.raises(TopologyGenerationError, match="Backbone integrity validation failed"):
        Amber().write_itp(top, "col", merged_pdb_file=str(p))
