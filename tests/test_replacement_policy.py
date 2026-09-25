from collections import Counter
from contextlib import nullcontext
from itertools import combinations
import asyncio
import json
import random
import shutil
from types import SimpleNamespace

import pytest

from colbuilder.core.geometry.crosslink_network import (
    CrosslinkIntegrityError, Entity, Network, after_replacement, refresh_network,
    replacement_survivors, select_replacements,
)
from colbuilder.core.geometry.replacement_policy import (
    model_incidence, validate_attachment_plan, validate_replacement_mode,
)
from colbuilder.core.topology.replacement_validation import validate_replacement_itps
from colbuilder.core.utils.config import ColbuilderConfig
from test_crosslink_network import terminal_network, long_pyd, write_markers


@pytest.mark.parametrize("seed", range(100))
def test_joint_pyd50_retains_one_complete_pyd_per_pair(seed):
    network = terminal_network()
    instructions, report = select_replacements(network, 50, "enzymatic", seed,
                                               mode="preserve_attachment")
    retained, _ = replacement_survivors(network, instructions)
    assert len(instructions) == 30
    assert len(retained) == 10
    assert Counter(e.scope for e in retained) == {"enzymatic_n": 5, "enzymatic_c": 5}
    assert model_incidence(retained) == Counter({i: 1 for i in range(20)})
    assert not report["attachment"]["newly_unlinked_models"]
    validate_attachment_plan(json.loads(json.dumps(report)), retained)


@pytest.mark.parametrize("ratio", [0, 10, 20, 30, 40, 50])
def test_feasible_ratios_have_exact_quotas(ratio):
    network = terminal_network()
    instructions, report = select_replacements(network, ratio, "all", 14, "preserve_attachment")
    retained, _ = replacement_survivors(network, instructions)
    assert len(instructions) == 6 * round(ratio / 10)
    assert all(t["removed"] == round(ratio / 10) for t in report["targets"].values())
    assert len(model_incidence(retained)) == 20


@pytest.mark.parametrize("ratio", [60, 80, 100])
def test_infeasible_quotas_fail_without_mutating_network(ratio):
    network = terminal_network()
    original = network.entities
    with pytest.raises(CrosslinkIntegrityError, match="At most 10/20"):
        select_replacements(network, ratio, "enzymatic", 1, "preserve_attachment")
    assert network.entities == original and network.replacement_report is None


def test_random_mode_keeps_legacy_seeded_draws():
    network = terminal_network()
    for seed in range(10):
        rng, expected = random.Random(seed), set()
        for scope in sorted({e.scope for e in network.entities}):
            bucket = sorted([e for e in network.entities if e.scope == scope], key=lambda e: e.identity)
            rng.shuffle(bucket)
            expected.update(e.identity for e in bucket[:5])
        instructions, report = select_replacements(network, 50, "enzymatic", seed)
        remaining, _ = replacement_survivors(network, instructions)
        assert {e.identity for e in network.entities} - {e.identity for e in remaining} == expected
        assert report["mode"] == "random"


def test_seed_reproducible_and_local():
    state = random.getstate()
    first = select_replacements(terminal_network(), 50, "enzymatic", 70, "preserve_attachment")
    second = select_replacements(terminal_network(), 50, "enzymatic", 70, "preserve_attachment")
    assert first == second
    assert first != select_replacements(terminal_network(), 50, "enzymatic", 71, "preserve_attachment")
    assert random.getstate() == state


def pair_entity(i, a, b, scope="non_enzymatic"):
    keys = ((a, str(200+i), "A", "LGX"), (b, str(400+i), "B", "AGS"))
    return Entity("GCP", scope, keys, ((keys[0] + ("CE",), keys[1] + ("NZ",)),))


def test_unselected_scope_can_keep_models_attached():
    base = terminal_network()
    age = tuple(pair_entity(i, 2*i, 2*i+1) for i in range(10))
    network = Network(base.markers, base.entities + age)
    _, report = select_replacements(network, 100, "enzymatic", 1, "preserve_attachment")
    assert report["removed_entities"] == 20
    assert len(report["retained_crosslinks"]) == 10
    assert not report["attachment"]["newly_unlinked_models"]
    assert len(select_replacements(network, 100, "non_enzymatic", 1)[0]) == 20


def test_degree_constraint_is_not_a_global_connected_component_constraint():
    entities = tuple(pair_entity(i, i, i+1) for i in range(3))
    network = Network({}, entities)
    _, report = select_replacements(network, 33, "all", 1, "preserve_attachment")
    assert report["removed_crosslinks"][0]["markers"] == entities[1].markers
    assert not report["attachment"]["newly_unlinked_models"]


def test_general_graph_solver_matches_exhaustive_feasibility():
    rng = random.Random(501)
    for _ in range(20):
        pairs = rng.sample(list(combinations(range(6), 2)), 7)
        entities = tuple(pair_entity(i, *pair, scope="enzymatic_n" if i < 3 else "enzymatic_c")
                         for i, pair in enumerate(pairs))
        network = Network({}, entities)
        initial = set(model_incidence(entities))
        feasible = any(
            set(model_incidence([e for i, e in enumerate(entities) if i not in set(a+b)])) == initial
            for a in combinations(range(3), 2) for b in combinations(range(3, 7), 2)
        )
        if feasible:
            _, report = select_replacements(network, 50, "all", 14, "preserve_attachment")
            assert not report["attachment"]["newly_unlinked_models"]
        else:
            with pytest.raises(CrosslinkIntegrityError, match="Infeasible"):
                select_replacements(network, 50, "all", 14, "preserve_attachment")


def test_empty_scope_and_invalid_mode():
    _, report = select_replacements(terminal_network(), 100, "non_enzymatic", 0, "preserve_attachment")
    assert not report["targets"] and report["removed_entities"] == 0
    with pytest.raises(CrosslinkIntegrityError, match="ratio_replace_mode"):
        select_replacements(terminal_network(), 50, "all", mode="invalid")


def test_config_defaults_and_manual_incompatibility():
    assert ColbuilderConfig(species="rattus_norvegicus").ratio_replace_mode == "random"
    for values in [{"ratio_replace_mode": "invalid"}, {"ratio_replace_seed": 1.5},
                   {"ratio_replace_mode": "preserve_attachment", "replace_bool": True},
                   {"ratio_replace_mode": "preserve_attachment", "replace_bool": True,
                    "ratio_replace": 50, "manual_replacements": ["0.caps.pdb LYS 5 B"]},
                   {"ratio_replace_mode": "preserve_attachment", "replace_bool": True,
                    "ratio_replace": 50, "topology_generator": True, "force_field": "martini3"}]:
        with pytest.raises(Exception):
            ColbuilderConfig(species="rattus_norvegicus", **values)
    with pytest.raises(CrosslinkIntegrityError, match="manual_replacements"):
        validate_replacement_mode(SimpleNamespace(ratio_replace_mode="preserve_attachment",
                                                 ratio_replace=50, manual_replacements=["x"]))


def test_contract_survives_refresh_and_rejects_additional_deletion(tmp_path):
    from colbuilder.core.geometry.crosslink_network import resolve_network
    caps, cfg = long_pyd(tmp_path)
    network = resolve_network(caps, cfg)
    _, network.replacement_report = select_replacements(network, 0, "all", 1, "preserve_attachment")
    final = after_replacement(network, [], caps)
    refreshed = refresh_network(final, caps)
    assert refreshed.replacement_report == network.replacement_report
    with pytest.raises(CrosslinkIntegrityError, match="lost all crosslinks"):
        validate_attachment_plan(network.replacement_report, [])
    with pytest.raises(CrosslinkIntegrityError, match="differ from"):
        validate_attachment_plan(network.replacement_report, network.entities*2)


@pytest.fixture
def final_itp(tmp_path):
    base = terminal_network()
    entity = base.entities[0]
    network = Network({key: base.markers[key] for key in entity.markers}, (entity,))
    _, network.replacement_report = select_replacements(network, 0, "all", 1, "preserve_attachment")
    path = tmp_path / "col_0_1.itp"
    text = ("[ moleculetype ]\ncol_0_1 3\n[ atoms ]\n"
            "1 CA 944 LYX C12 1 0 12\n2 CA 944 LYX C13 1 0 12\n"
            "3 CT 944 LY2 CB 2 0 12\n4 CT 944 LY3 CG 3 0 12\n"
            "[ bonds ]\n1 2 1\n1 3 1\n2 4 1\n")
    path.write_text(text)
    source_map = {key: i for i, key in enumerate(entity.markers)}
    return network, path, source_map, text


def test_final_itp_validation_reads_actual_bonds(final_itp):
    network, path, mapping, text = final_itp
    result = validate_replacement_itps(network, [(path, mapping)])
    assert result["degree_from_itp_bonds"] == {"0": 1, "1": 1}
    assert result["observed_intermodel_bonds"] == 2
    path.write_text(text.replace("2 4 1\n", ""))
    with pytest.raises(CrosslinkIntegrityError, match="missing"):
        validate_replacement_itps(network, [(path, mapping)])


@pytest.mark.parametrize("corruption", ["duplicate", "wrong", "empty", "reorder", "invalid_map"])
def test_final_itp_rejects_corruption(final_itp, corruption):
    network, path, mapping, text = final_itp
    if corruption == "duplicate":
        text += "1 3 1\n"
    elif corruption == "wrong":
        text = text.replace("1 3 1\n", "2 3 1\n")
    elif corruption == "empty":
        text = "[ atoms ]\n"
    elif corruption == "reorder":
        text = text.replace("944 LY2", "945 LY2")
    else:
        mapping = {key: 0 for key in mapping}
    path.write_text(text)
    with pytest.raises(CrosslinkIntegrityError):
        validate_replacement_itps(network, [(path, mapping)])


def test_final_itp_rejects_missing_or_duplicated_groups(final_itp):
    network, path, mapping, _ = final_itp
    for groups in [[], [(path, mapping), (path, mapping)], [(path.with_name("missing.itp"), mapping)]]:
        with pytest.raises(CrosslinkIntegrityError):
            validate_replacement_itps(network, groups)


def test_random_mode_does_not_claim_attachment_validation(final_itp):
    network, path, mapping, _ = final_itp
    network.replacement_report["mode"] = "random"
    assert validate_replacement_itps(network, [(path, mapping)]) is None


def test_three_model_pyd_is_one_entity_with_three_protected_models():
    base = terminal_network()
    original = base.entities[0]
    keys = (original.markers[0], original.markers[1], (14, "9", "C", "LY3"))
    entity = Entity("PYD", "enzymatic_n", keys,
                    ((keys[0]+("C12",), keys[1]+("CB",)),
                     (keys[0]+("C13",), keys[2]+("CG",))))
    _, report = select_replacements(Network({}, (entity,)), 0, "all", 1, "preserve_attachment")
    assert report["attachment"]["protected_models"] == [0, 1, 14]
    with pytest.raises(CrosslinkIntegrityError, match="Infeasible"):
        select_replacements(Network({}, (entity,)), 100, "all", 1, "preserve_attachment")


def test_insertion_codes_use_explicit_source_numbers(final_itp):
    from dataclasses import replace
    network, path, mapping, _ = final_itp
    entity = network.entities[0]
    keys = tuple((m[0], m[1]+"A", m[2], m[3]) for m in entity.markers)
    bonds = tuple(tuple((p[0], p[1]+"A", *p[2:]) for p in bond) for bond in entity.bonds)
    network.entities = (replace(entity, markers=keys, bonds=bonds),)
    _, network.replacement_report = select_replacements(network, 0, "all", 1, "preserve_attachment")
    result = validate_replacement_itps(network, [(path, dict(zip(keys, range(3))), ["944"]*3)])
    assert result["observed_complete_crosslinks"] == 1


@pytest.mark.parametrize("ratio", [0, 50, 75])
def test_automatic_orphan_cleanup_does_not_override_joint_ratio(tmp_path, monkeypatch, ratio):
    """Exercise stage ordering; native Chimera/backbone are covered by the saved-caps regression."""
    from pathlib import Path
    import colbuilder.core.geometry.geometry_replacer as module

    geometry = tmp_path / ".tmp/geometry_gen/T"
    for i in range(2):
        centers, arms = [], []
        for number, offset in [(944, 0), (103, 20)]:
            x = i*100 + offset
            centers.append(("LYX", number, "B", {"C12": (x, 0, 0), "C13": (x, 2, 0)}))
            arms.extend([("LY2", number, "A", {"CB": (x+1.5, 0, 0)}),
                         ("LY3", number, "C", {"CG": (x+1.5, 2, 0)})])
        if i == 0:
            centers.append(("LGX", 555, "A", {"CE": (500, 0, 0)}))
        write_markers(geometry / f"{2*i}.caps.pdb", centers)
        write_markers(geometry / f"{2*i+1}.caps.pdb", arms)
    models = {i: SimpleNamespace(type="T", connect=[i]) for i in range(4)}
    system = SimpleNamespace(get_models=lambda: list(models),
                             get_model=lambda model_id: models[model_id])
    cfg = SimpleNamespace(working_directory=tmp_path, auto_fix_unpaired=True,
                          _auto_unpaired_replacements=["0.caps.pdb LYS 555 A"],
                          manual_replacements=None, ratio_replace=ratio,
                          ratio_replace_mode="preserve_attachment", ratio_replace_seed=14,
                          ratio_replace_scope="enzymatic",
                          n_term_combination="944.B - 944.A - 944.C",
                          c_term_combination="103.B - 103.A - 103.C")
    calls = []

    def mutate(root, instructions):
        for instruction in instructions:
            name, target, resid, chain = instruction.split()
            path = root / name
            path.write_text("".join(
                line[:17]+target+line[20:]
                if line.startswith("ATOM") and line[21] == chain and line[22:26].strip() == resid
                else line for line in path.read_text().splitlines(True)))

    async def cleanup(self, source_dir, dest_dir, manual_list, **kwargs):
        calls.append("cleanup")
        shutil.copytree(source_dir / "T", dest_dir / "T")
        mutate(dest_dir / "T", manual_list)

    async def chimera(self, config, instructions, type_dir, working_dir):
        calls.append("ratio")
        mutate(type_dir, Path(instructions).read_text().splitlines())
        return True

    monkeypatch.setattr(module.CrosslinkReplacer, "_apply_manual_replacements_to_dir", cleanup)
    monkeypatch.setattr(module.CrosslinkReplacer, "_run_chimera_command", chimera)
    monkeypatch.setattr(module, "preserve_backbone", lambda *args: nullcontext())
    if ratio == 75:
        with pytest.raises(Exception, match="Infeasible"):
            asyncio.run(module.CrosslinkReplacer().replace_in_system(system, cfg))
        assert calls == ["cleanup"]
    else:
        asyncio.run(module.CrosslinkReplacer().replace_in_system(system, cfg))
        assert calls == (["cleanup", "ratio"] if ratio else ["cleanup"])
        assert len(system.crosslink_network.entities) == (2 if ratio else 4)
        assert not system.crosslink_network.replacement_report["attachment"]["newly_unlinked_models"]
        assert all(key[3] != "LGX" for key in system.crosslink_network.markers)
        assert not getattr(cfg, "_replacement_skipped", False)
