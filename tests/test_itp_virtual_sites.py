from pathlib import Path

from colbuilder.core.topology.itp import Itp


class _TwoConnectionModel:
    connect = (1, 2)


class _TwoConnectionSystem:
    def get_model(self, model_id):
        return _TwoConnectionModel()


def _component_itp(residue: str) -> str:
    atom_rows = []
    for atom_id in range(1, 15):
        fields = [
            str(atom_id),
            "P4",
            "1",
            residue,
            f"B{atom_id}",
            str(atom_id),
            "0.000",
            "72.000",
        ]
        if atom_id == 1:
            fields.extend(("P5", "0.250", "73.500"))
        atom_rows.append("\t".join(fields))

    return "\n".join(
        (
            "[ moleculetype ]",
            f"{residue} 1",
            "",
            "[ atoms ]",
            *atom_rows,
            "",
            "[ bonds ]",
            "1 2 1 0.334 32117.450",
            "",
            "[virtual_sites2]",
            "11\t1\t2\t1\t0.500000",
            "",
            "[ virtual_sites3 ]",
            "12 2 3 4 4 0.100000 -0.200000 3.000000",
            "",
            "[\tvirtual_sites4\t]",
            "13 1 2 3 4 2 0.125000 0.250000 0.375000",
            "",
            "[ virtual_sitesn ]",
            "; A general virtual_sitesn row and a one-constructor Go row",
            "14 1 5 6 7",
            "10 1 8",
            "",
        )
    )


def _section_rows(path: Path, section: str):
    rows = []
    current_section = None
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current_section = stripped[1:-1].strip()
        elif (
            current_section == section
            and stripped
            and not stripped.startswith((";", "#"))
        ):
            rows.append(stripped.split())
    return rows


def test_read_keeps_all_atom_and_virtual_site_fields(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "col_7.1.itp").write_text(_component_itp("ONE"))

    topology = Itp(system=_TwoConnectionSystem(), model_id=7)
    topology.read_itp(model_id=7, connect_id=1, cnt_con=0)

    assert topology.atoms[0][0] == [
        "1",
        "P4",
        "1",
        "ONE",
        "B1",
        "1",
        "0.000",
        "72.000",
        "P5",
        "0.250",
        "73.500",
    ]
    assert topology.virtual_sites2[0] == [["11", "1", "2", "1", "0.500000"]]
    assert topology.virtual_sites3[0] == [
        ["12", "2", "3", "4", "4", "0.100000", "-0.200000", "3.000000"]
    ]
    assert topology.virtual_sites4[0] == [
        ["13", "1", "2", "3", "4", "2", "0.125000", "0.250000", "0.375000"]
    ]
    assert topology.virtual_sitesn[0] == [
        ["14", "1", "5", "6", "7"],
        ["10", "1", "8"],
    ]
    assert topology.virtual_sites is topology.virtual_sitesn


def test_merge_and_write_offsets_only_virtual_site_atom_columns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "col_7.1.itp").write_text(_component_itp("ONE"))
    (tmp_path / "col_7.2.itp").write_text(_component_itp("TWO"))

    topology = Itp(system=_TwoConnectionSystem(), model_id=7)
    topology.read_itp(model_id=7, connect_id=1, cnt_con=0)
    topology.read_itp(model_id=7, connect_id=2, cnt_con=1)
    topology.merge_topology(cnt_con=0)
    topology.merge_topology(cnt_con=1)
    topology.write_topology(cnt_model=7)

    output = tmp_path / "col_7.itp"
    atom_rows = _section_rows(output, "atoms")
    assert atom_rows[0] == [
        "1",
        "P4",
        "1",
        "ONE",
        "B1",
        "1",
        "0.000",
        "72.000",
        "P5",
        "0.250",
        "73.500",
    ]
    assert atom_rows[14] == [
        "15",
        "P4",
        "1",
        "TWO",
        "B1",
        "15",
        "0.000",
        "72.000",
        "P5",
        "0.250",
        "73.500",
    ]

    assert _section_rows(output, "virtual_sites2") == [
        ["11", "1", "2", "1", "0.500000"],
        ["25", "15", "16", "1", "0.500000"],
    ]
    assert _section_rows(output, "virtual_sites3") == [
        ["12", "2", "3", "4", "4", "0.100000", "-0.200000", "3.000000"],
        ["26", "16", "17", "18", "4", "0.100000", "-0.200000", "3.000000"],
    ]
    assert _section_rows(output, "virtual_sites4") == [
        ["13", "1", "2", "3", "4", "2", "0.125000", "0.250000", "0.375000"],
        ["27", "15", "16", "17", "18", "2", "0.125000", "0.250000", "0.375000"],
    ]
    assert _section_rows(output, "virtual_sitesn") == [
        ["14", "1", "5", "6", "7"],
        ["10", "1", "8"],
        ["28", "1", "19", "20", "21"],
        ["24", "1", "22"],
    ]
    assert topology.final_virtual_sites is topology.final_virtual_sitesn


def test_decimal_g21_bond_force_constant_is_not_treated_as_flexible(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "col_7.1.itp").write_text(_component_itp("ONE"))

    topology = Itp(system=_TwoConnectionSystem(), model_id=7)
    topology.read_itp(model_id=7, connect_id=1, cnt_con=0)
    topology.merge_topology(cnt_con=0)

    assert topology.final_bonds == [[1, 2, "1", "0.334", "32117.450\n"]]
    assert topology.final_flex_bonds == []
