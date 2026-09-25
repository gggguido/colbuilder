from collections import Counter
from contextlib import nullcontext
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

from colbuilder.core.geometry.backbone import BACKBONE, read_pdb_residues
from colbuilder.core.geometry.crosslink_network import (
    CrosslinkIntegrityError, Entity, Network, active_caps, distance, resolve_network,
)
from colbuilder.core.geometry.crosslink_mixer import CrosslinkMixer
from colbuilder.core.geometry.shared_site_mixing import (
    OUTPUT_TYPE, _render, _rigid_fit, choose_alternatives, compose_shared_sites, match_loci,
)
from colbuilder.core.geometry.unpaired_crosslinks import UnpairedCrosslinkFinder
from colbuilder.core.utils.config import ColbuilderConfig
from test_backbone_preservation import atom


class System:
    def __init__(self, ids):
        self.models = {mid: SimpleNamespace(type="A", connect=[mid]) for mid in ids}

    def get_models(self):
        return list(self.models)

    def get_size(self):
        return len(self.models)

    def get_model(self, model_id):
        return self.models[model_id]


def make_variants(tmp_path, n=2, extras=True, common=False):
    records = {label: {mid: [] for mid in range(2 * n)} for label in ("A", "B")}

    def add(mid, number, chain, names, ports, origin):
        for label in records:
            name = names[label]
            xyz = np.asarray(origin, dtype=float)
            atoms = {key: tuple(xyz + delta) for key, delta in {
                "N": (0, 4, 0), "CA": (1, 4, 0), "C": (1, 5, 0), "O": (2, 5, 0),
            }.items()}
            atoms.update({key: tuple(xyz + delta) for key, delta in ports[label].items()})
            records[label][mid].append((name, number, chain, atoms))

    for i in range(n):
        for end, number in enumerate((944, 103)):
            origin = (i * 100, end * 30, 0)
            add(2*i, number, "B", {"A": "LYX", "B": "LZD"},
                {"A": {"C12": (0, 0, 0), "C13": (0, 1.4, 0)}, "B": {"CE": (0, 1.4, 0)}}, origin)
            add(2*i+1, number, "A", {"A": "LY2", "B": "LYS"},
                {"A": {"CB": (1.5, 0, 0)}, "B": {"CB": (1.5, 0, 0)}}, origin)
            add(2*i+1, number, "C", {"A": "LY3", "B": "LZS"},
                {"A": {"CG": (1.5, 1.4, 0)}, "B": {"NZ1": (1.5, 1.4, 0)}}, origin)
    if extras:
        # This extra edge connects independently mixed terminal components.
        partner = 2 if n > 1 else 1
        add(0, 523, "A", {"A": "LGX" if common else "LYS", "B": "LGX"},
            {"A": {"CE": (0, 0, 0)}, "B": {"CE": (0, 0, 0)}}, (0, 80, 0))
        add(partner, 286, "C", {"A": "AGS" if common else "ARG", "B": "AGS"},
            {"A": {"NZ" if common else "NE": (1.5, 0, 0)}, "B": {"NZ": (1.5, 0, 0)}}, (0, 80, 0))
    variants = {}
    for label, models in records.items():
        variants[label] = {}
        for mid, residues in models.items():
            path = tmp_path / label / f"{mid}.caps.pdb"
            path.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            serial = 0
            for name, number, chain, atoms in residues:
                for key, xyz in atoms.items():
                    serial += 1
                    row = atom(serial, key, name, number, chain=chain, xyz=xyz)
                    lines.append(row[:12] + f" {key:<3}" + row[16:])
                lines.append("TER\n")
            path.write_text("".join(lines) + "END\n")
            variants[label][mid] = path
    return variants, System(variants["A"])


def test_match_complete_pyd_mold_and_additional_gcp(tmp_path):
    variants, _ = make_variants(tmp_path)
    common, alternatives, extra = match_loci({label: resolve_network(caps) for label, caps in variants.items()})
    assert len(alternatives) == 4
    assert not common
    assert len(extra) == 1
    assert extra[0].options["B"].kind == "GCP"
    assert {len(locus.positions) for locus in alternatives} == {3}


@pytest.mark.parametrize("ratio,expected", [(0, 0), (25, 2), (50, 2), (75, 4), (100, 4)])
def test_composition_preserves_union_and_backbones(tmp_path, ratio, expected):
    variants, system = make_variants(tmp_path / "input")
    original = {p: p.read_bytes() for caps in variants.values() for p in caps.values()}
    state = random.getstate()
    report = compose_shared_sites(system, variants, {"A": ratio, "B": 100-ratio}, tmp_path / "out")
    assert random.getstate() == state
    counts = Counter(e.kind for e in system.crosslink_network.entities)
    assert counts == Counter(PYD=expected, MOLD=4-expected, GCP=1)
    assert all(f["target"] == f["achieved"] for f in report["families"])
    assert report["additional_loci"] == 1
    selected = active_caps(system, tmp_path / "out")
    assert len(system.crosslink_network.markers) == expected * 3 + (4-expected) * 2 + 2
    for mid, path in selected.items():
        source = variants[report["model_sources"][mid]][mid]
        before = read_pdb_residues(source.read_text().splitlines(True), str(source))
        after = read_pdb_residues(path.read_text().splitlines(True), str(path))
        assert [(r.address, r.segment) for r in before] == [(r.address, r.segment) for r in after]
        for a, b in zip(before, after):
            for name in BACKBONE & a.atoms.keys():
                assert a.atoms[name][30:54] == b.atoms[name][30:54]
        assert system.get_model(mid).type == OUTPUT_TYPE
    assert all(path.read_bytes() == content for path, content in original.items())
    assert {e.identity for e in resolve_network(selected).entities} == {
        e.identity for e in system.crosslink_network.entities
    }
    assert system.get_model(0).connect == [0, 1, 2, 3]
    saved = json.loads((tmp_path / "out/crosslink_mix_report.json").read_text())
    assert saved["strategy"] == "shared_sites"


def test_common_locus_retained_once(tmp_path):
    variants, system = make_variants(tmp_path / "input", common=True)
    report = compose_shared_sites(system, variants, {"A": 50, "B": 50}, tmp_path / "out")
    assert report["common_loci"] == 1
    assert report["additional_loci"] == 0
    assert sum(e.kind == "GCP" for e in system.crosslink_network.entities) == 1


def test_no_alternatives_retains_union_even_zero_weight_donor(tmp_path):
    variants, system = make_variants(tmp_path / "input")
    # A second GCP-only difference, keeping the terminal PYDs identical.
    for mid, path in variants["B"].items():
        b = read_pdb_residues(path.read_text().splitlines(True), str(path))
        a = read_pdb_residues(variants["A"][mid].read_text().splitlines(True), "A")
        blocks = []
        for ra, rb in zip(a, b):
            residue = rb if rb.name in {"LGX", "AGS"} else ra
            blocks.extend(residue.atoms.values())
            blocks.append("TER\n")
        path.write_text("".join(blocks) + "END\n")
    report = compose_shared_sites(system, variants, {"A": 100, "B": 0}, tmp_path / "out")
    assert not report["families"]
    assert report["common_loci"] == 4
    assert report["counts"] == {"PYD:enzymatic": 4, "GCP:non_enzymatic": 1}


def test_ambiguous_partial_site_conflict_fails():
    pyd = Entity("PYD", "enzymatic", ((0, "1", "A", "LYX"), (1, "2", "B", "LY2"), (1, "3", "C", "LY3")), ())
    partial = Entity("GCP", "non_enzymatic", ((0, "1", "A", "LGX"), (3, "4", "C", "AGS")), ())
    with pytest.raises(CrosslinkIntegrityError, match="overlapping"):
        match_loci({"A": Network({}, (pyd,)), "B": Network({}, (partial,))})


def test_ambiguous_multiple_site_matches_fail():
    pyd = Entity("PYD", "enzymatic", ((0, "1", "A", "LYX"), (1, "2", "B", "LY2"), (1, "3", "C", "LY3")), ())
    x = Entity("MOLD", "non_enzymatic", ((0, "1", "A", "LZD"), (1, "3", "C", "LZS")), ())
    y = Entity("GCP", "non_enzymatic", ((1, "2", "B", "LGX"), (4, "7", "C", "AGS")), ())
    with pytest.raises(CrosslinkIntegrityError, match="Ambiguous"):
        match_loci({"A": Network({}, (pyd,)), "B": Network({}, (x, y))})


def test_infeasible_quota_does_not_silently_overshoot(tmp_path):
    variants, _ = make_variants(tmp_path)
    _, loci, _ = match_loci({label: resolve_network(caps) for label, caps in variants.items()})
    # Put both copies in a single indivisible component.
    for locus in loci:
        locus.options = {label: Entity(e.kind, e.scope, tuple((0,) + k[1:] if k[0] == 2 else k for k in e.markers), e.bonds)
                         for label, e in locus.options.items()}
    with pytest.raises(CrosslinkIntegrityError, match="cannot be realized"):
        choose_alternatives(loci, {"A": 50, "B": 50})


@pytest.mark.parametrize("ratios", [{"A": -1, "B": 101}, {"A": 10, "B": 10}, {"A": float("nan"), "B": 0}])
def test_invalid_ratios_fail(ratios):
    with pytest.raises(CrosslinkIntegrityError, match="percentages"):
        choose_alternatives([], ratios)


def test_rigid_fit_preserves_handedness_and_lengths():
    source = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    target = source @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) + [7, 2, 3]
    rotation, translation, residual = _rigid_fit(source, target, .001, "test")
    assert np.linalg.det(rotation) == pytest.approx(1)
    assert residual < 1e-10
    assert np.allclose(source @ rotation + translation, target)
    with pytest.raises(CrosslinkIntegrityError, match="deforming"):
        _rigid_fit(source, source * 2, .001, "test")


def test_bad_alignment_rolls_back(tmp_path):
    variants, system = make_variants(tmp_path / "input")
    path = variants["A"][0]
    lines = path.read_text().splitlines(True)
    path.write_text("".join(line[:30] + f"{float(line[30:38])+5:8.3f}" + line[38:]
                            if line.startswith("ATOM") and line[17:20] == "LYS" and line[12:16].strip() in BACKBONE
                            else line for line in lines))
    with pytest.raises(CrosslinkIntegrityError, match="alignment residual"):
        compose_shared_sites(system, variants, {"A": 50, "B": 50}, tmp_path / "out")
    assert not (tmp_path / "out" / OUTPUT_TYPE).exists()
    assert not hasattr(system, "crosslink_network")
    assert all(m.type == "A" for m in system.models.values())


def test_entity_transfer_keeps_forming_distance_and_source_sidechain_geometry(tmp_path):
    variants, system = make_variants(tmp_path / "input")
    for path in variants["B"].values():
        lines = path.read_text().splitlines(True)
        path.write_text("".join(line[:30] + f"{float(line[30:38])+0.1:8.3f}" + line[38:]
                                if line.startswith("ATOM") and line[17:20] in {"AGS", "LGX"}
                                else line for line in lines))
    donor = resolve_network(variants["B"])
    report = compose_shared_sites(system, variants, {"A": 50, "B": 50}, tmp_path / "out")
    assert 0 < report["transfers"][0]["max_residual_A"] < .25
    entity = next(e for e in donor.entities if e.kind == "GCP")
    a, b = entity.bonds[0]
    assert distance(donor.markers, a, b) == pytest.approx(distance(system.crosslink_network.markers, a, b), abs=.002)


def test_native_sequence_mismatch_is_not_silently_accepted(tmp_path):
    variants, system = make_variants(tmp_path / "input")
    path = variants["A"][2]
    path.write_text(path.read_text().replace("ARG", "LYS"))
    with pytest.raises(CrosslinkIntegrityError, match="native residues"):
        compose_shared_sites(system, variants, {"A": 50, "B": 50}, tmp_path / "out")


def test_serial_dependent_records_are_remapped_without_inventing_chain_breaks():
    lines = [atom(1, "N", "LYS", 1), atom(2, "CA", "LYS", 1), atom(3, "C", "LYS", 1),
             atom(4, "O", "LYS", 1), atom(5, "CB", "LYS", 1), "TER\n",
             atom(7, "N", "GLY", 2), "TER\n", "CONECT    1    2    5\n",
             "CONECT    5    1\n", "END\n"]
    residues = read_pdb_residues(lines, "input")
    replacement = [line[:17] + "LGX" + line[20:] for line in lines[:4]]
    replacement.extend([atom(8, "CE", "LGX", 1), atom(9, "CB", "LGX", 1)])
    result = _render(lines, residues, {residues[0].address: replacement})
    assert "CONECT    1    2\n" in result
    assert "CONECT    5" not in result
    assert result.endswith("END\n")
    after = read_pdb_residues(result.splitlines(True), "output")
    assert [r.segment for r in after] == [r.segment for r in residues]
    assert [r.name for r in after] == ["LGX", "GLY"]


def test_named_ter_keeps_boundary_and_updated_residue_identity():
    rows = [atom(i, name, "LYS", 1) for i, name in enumerate(("N", "CA", "C", "O", "CB"), 1)]
    rows.extend(["TER       6      LYS C   1 \n", "END\n"])
    residues = read_pdb_residues(rows, "input")
    block = [row[:17] + "LGX" + row[20:] for row in rows[:5]]
    result = _render(rows, residues, {residues[0].address: block})
    end = next(row for row in result.splitlines() if row.startswith("TER"))
    assert end[17:20] == "LGX"
    assert end[21:27] == rows[0][21:27]


def test_scaffold_mismatch_fails(tmp_path):
    variants, system = make_variants(tmp_path / "input")
    path = variants["B"][0]
    path.write_text(path.read_text().replace("TER\n", "", 1))
    with pytest.raises(CrosslinkIntegrityError, match="TER"):
        compose_shared_sites(system, variants, {"A": 50, "B": 50}, tmp_path / "out")


def test_missing_model_fails(tmp_path):
    variants, system = make_variants(tmp_path / "input")
    del variants["B"][3]
    with pytest.raises(CrosslinkIntegrityError):
        compose_shared_sites(system, variants, {"A": 50, "B": 50}, tmp_path / "out")


def test_no_overwrite_of_composed_caps(tmp_path):
    variants, system = make_variants(tmp_path / "input")
    compose_shared_sites(system, variants, {"A": 50, "B": 50}, tmp_path / "out")
    with pytest.raises(CrosslinkIntegrityError, match="overwrite"):
        compose_shared_sites(system, variants, {"A": 50, "B": 50}, tmp_path / "out")


def test_config_strategy_defaults_and_validation(tmp_path):
    assert ColbuilderConfig(species="rattus_norvegicus").mix_strategy == "whole_models"
    variants, _ = make_variants(tmp_path)
    files = [str(variants[label][0]) for label in variants]
    data = dict(species="rattus_norvegicus", mix_bool=True, files_mix=files, ratio_mix="A:50 B:50", mix_strategy="shared_sites")
    assert ColbuilderConfig(**data).mix_alignment_tolerance == .25
    for changes in [{"mix_strategy": "unknown"}, {"mix_alignment_tolerance": 0},
                    {"ratio_mix": "A:100"}, {"ratio_mix": "../A:50 B:50"},
                    {"ratio_mix": "_mixed_sites:50 B:50"}]:
        with pytest.raises(Exception):
            ColbuilderConfig(**(data | changes))


def test_finder_only_reads_explicit_active_variants(tmp_path):
    variants, _ = make_variants(tmp_path)
    # Removing LGX from the selected A caps leaves only AGS in selected B.
    selected = {0: variants["A"][0], 1: variants["A"][1], 2: variants["B"][2], 3: variants["B"][3]}
    finder = UnpairedCrosslinkFinder(tmp_path, geom_dir=tmp_path, selected_caps=selected)
    entries, _ = finder.run()
    assert entries == ["2.caps.pdb ARG 286 C"]


def test_native_arg_target_is_not_converted_to_lys(tmp_path, monkeypatch):
    from colbuilder.core.geometry import crosslink_mixer as module

    seen = []
    monkeypatch.setattr(module, "preserve_backbone", lambda directory, instructions: seen.extend(instructions) or nullcontext())
    monkeypatch.setattr(module, "Chimera", lambda *a, **kw: SimpleNamespace(swapaa=lambda **kw: SimpleNamespace(returncode=0)))
    assert CrosslinkMixer()._apply_chimera_swaps(["1.caps.pdb ARG 310 C", "2.caps.pdb LYS 523 A"], tmp_path, config=SimpleNamespace())
    assert seen == ["1.caps.pdb ARG 310 C", "2.caps.pdb LYS 523 A"]
    assert not CrosslinkMixer()._apply_chimera_swaps(["1.caps.pdb AGS 310 C"], tmp_path, config=SimpleNamespace())


def test_production_strategy_dispatch(tmp_path):
    variants, system = make_variants(tmp_path / "input")
    cfg = SimpleNamespace(mix_strategy="shared_sites", ratio_mix={"A": 50, "B": 50})
    report = CrosslinkMixer()._mix_generated_variants(system, cfg, tmp_path / "out", variants)
    assert report["counts"] == {"PYD:enzymatic": 2, "MOLD:non_enzymatic": 2, "GCP:non_enzymatic": 1}


def test_legacy_cleanup_is_routed_to_actual_file_owner(tmp_path, monkeypatch):
    from colbuilder.core.geometry import crosslink_mixer as module

    variants, system = make_variants(tmp_path)
    system.get_model(2).type = system.get_model(3).type = "B"
    monkeypatch.setattr(module.Mix, "add_mix", lambda self, system: system)
    monkeypatch.setattr(module.Connect, "write_connect", lambda *a, **kw: None)
    seen = []
    monkeypatch.setattr(CrosslinkMixer, "_apply_chimera_swaps", lambda self, instructions, directory, **kw: seen.append((instructions, directory)) or True)
    CrosslinkMixer()._mix_generated_variants(system, SimpleNamespace(ratio_mix={"A": 50, "B": 50}), tmp_path, variants)
    assert seen == [(["2.caps.pdb ARG 286 C"], tmp_path / "B")]


@pytest.mark.parametrize("ratio", [0, 30, 50, 100])
def test_legacy_component_assignment_is_unchanged(tmp_path, monkeypatch, ratio):
    from copy import deepcopy
    from colbuilder.core.geometry import crosslink_mixer as module

    variants, system = make_variants(tmp_path, n=4, common=True)
    for mid in system.get_models():
        system.get_model(mid).connect = [mid // 2 * 2, mid // 2 * 2 + 1]
    reference = deepcopy(system)
    ratios = {"A": ratio, "B": 100-ratio}
    module.Mix(ratio_mix=ratios, system=reference).add_mix(system=reference)
    monkeypatch.setattr(module.Connect, "write_connect", lambda *a, **kw: None)
    cfg = SimpleNamespace(mix_strategy="whole_models", ratio_mix=ratios)
    CrosslinkMixer()._mix_generated_variants(system, cfg, tmp_path, variants)
    assert {mid: model.type for mid, model in system.models.items()} == {
        mid: model.type for mid, model in reference.models.items()
    }
    assert not (tmp_path / OUTPUT_TYPE).exists()
