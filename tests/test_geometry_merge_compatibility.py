"""Regression coverage for upstream/local replacement precedence integration."""

import asyncio
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

import colbuilder.core.geometry.main_geometry as geometry
from colbuilder.core.utils.config import ColbuilderConfig


ORPHAN_REPLACEMENTS = ["0.caps.pdb ARG 555 A"]


class MinimalSystem:
    def __init__(self):
        self.model = SimpleNamespace(type="T")

    def get_model(self, model_id):
        assert model_id == 0
        return self.model

    def write_pdb(self, pdb_out, **kwargs):
        Path(pdb_out).write_text("REMARK minimal geometry service fixture\nEND\n")


class MinimalFiles:
    def __init__(self, root):
        self.geometry_dir = root / ".tmp" / "geometry_gen"
        self.replacement_dir = root / ".tmp" / "replace_crosslinks"
        self.output_dir = root / "output"

    def ensure_geometry_dir(self):
        self.geometry_dir.mkdir(parents=True, exist_ok=True)
        return self.geometry_dir

    def ensure_replacement_dir(self):
        self.replacement_dir.mkdir(parents=True, exist_ok=True)
        return self.replacement_dir

    def copy_to_output(self, source):
        self.output_dir.mkdir(exist_ok=True)
        destination = self.output_dir / source.name
        shutil.copy2(source, destination)
        return destination


@pytest.fixture
def run_generation(tmp_path, monkeypatch):
    """Exercise the service orchestration without invoking Chimera or a lattice."""
    system = MinimalSystem()
    events, replacement_configs = [], []

    class Builder:
        def set_file_manager(self, manager):
            self.file_manager = manager

        async def build(self, config):
            events.append("build")
            return system

    class Finder:
        def __init__(self, base_dir):
            assert base_dir == tmp_path

        def run(self):
            events.append("find_orphans")
            return list(ORPHAN_REPLACEMENTS), None

    class Replacer:
        async def replace_in_system(self, received_system, config, directory):
            assert received_system is system
            assert config.working_directory == tmp_path
            events.append("replace")
            replacement_configs.append(config)
            return received_system

    monkeypatch.setattr(geometry, "CrystalBuilder", Builder)
    monkeypatch.setattr(geometry, "UnpairedCrosslinkFinder", Finder)
    monkeypatch.setattr(geometry, "CrosslinkReplacer", Replacer)

    def run(**options):
        # Configuration validation is tested separately. Keep a real Pydantic
        # model here so copying and private cleanup metadata match production.
        config = ColbuilderConfig.model_construct(
            working_directory=tmp_path,
            geometry_generator=True,
            mix_bool=False,
            pdb_file=None,
            output="generated",
            species="homo_sapiens",
            fibril_length=40,
            debug=False,
            **options,
        )
        service = geometry.GeometryService(config, MinimalFiles(tmp_path))
        original_directory = Path.cwd()
        returned_system, output = asyncio.run(service._handle_full_generation())
        assert Path.cwd() == original_directory
        assert returned_system is system
        assert output.is_file()
        assert events == ["build", "find_orphans", "replace"]
        assert len(replacement_configs) == 1
        return config, replacement_configs[0]

    return run


@pytest.mark.parametrize(
    "explicit_request",
    [
        {"ratio_replace": 50},
        {"manual_replacements": ["0.caps.pdb LYS 944 B"]},
        {"replace_file": Path("explicit_replacements.pdb")},
    ],
    ids=["ratio", "manual", "replace-file"],
)
def test_explicit_random_replacement_is_not_overridden_by_orphans(run_generation, explicit_request):
    original, forwarded = run_generation(
        replace_bool=True, ratio_replace_mode="random", **explicit_request
    )

    for field in ("ratio_replace", "manual_replacements", "replace_file"):
        assert getattr(forwarded, field) == getattr(original, field)
    assert not forwarded.auto_fix_unpaired
    assert not original.auto_fix_unpaired
    assert not hasattr(forwarded, "_auto_unpaired_replacements")


@pytest.mark.parametrize("ratio", [0, 50])
def test_attachment_cleanup_keeps_ratio_and_uses_private_instructions(run_generation, ratio):
    original, forwarded = run_generation(
        replace_bool=True,
        ratio_replace_mode="preserve_attachment",
        ratio_replace=ratio,
        ratio_replace_seed=14,
        manual_replacements=None,
    )

    assert forwarded._auto_unpaired_replacements == ORPHAN_REPLACEMENTS
    assert forwarded.manual_replacements is None
    assert forwarded.ratio_replace == original.ratio_replace == ratio
    assert forwarded.ratio_replace_seed == 14
    assert forwarded.replace_bool
    assert forwarded.auto_fix_unpaired
    assert original.auto_fix_unpaired
    assert not hasattr(original, "_auto_unpaired_replacements")


def test_no_explicit_replacement_enables_orphan_cleanup(run_generation):
    original, forwarded = run_generation(
        replace_bool=False,
        ratio_replace_mode="random",
        ratio_replace=None,
        manual_replacements=None,
        replace_file=None,
    )

    assert forwarded.manual_replacements == ORPHAN_REPLACEMENTS
    assert forwarded.ratio_replace == 0
    assert forwarded.replace_bool
    assert forwarded.auto_fix_unpaired
    assert original.auto_fix_unpaired
    assert original.manual_replacements is None
    assert original.ratio_replace is None
    assert not original.replace_bool
    assert not hasattr(forwarded, "_auto_unpaired_replacements")
