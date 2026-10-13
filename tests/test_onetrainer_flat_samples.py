"""Training previews land loose in the run's samples folder, not one folder per
prompt: the launcher rewrites where OneTrainer saves each one."""
import importlib.util
import sys
import types
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "core" / "engines" / "onetrainer" / "onetrainer_train.py"
_spec = importlib.util.spec_from_file_location("onetrainer_train", _SCRIPT)
launcher = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launcher)


def test_a_prompt_folder_preview_moves_up_with_its_index_in_the_name():
    dest = str(Path("run", "workspace", "samples", "2 - Ann reads a book",
                    "2026-10-13_05-40-45-training-sample-48-2-0"))
    assert launcher.flat_destination(dest) == str(Path(
        "run", "workspace", "samples", "2026-10-13_05-40-45-training-sample-48-2-0-p2"))


def test_anything_else_is_left_where_onetrainer_put_it():
    custom = str(Path("run", "workspace", "samples", "custom", "x"))
    other = str(Path("run", "elsewhere", "0 - Ann", "x"))
    assert launcher.flat_destination(custom) == custom
    assert launcher.flat_destination(other) == other


def test_the_hook_rewrites_the_destination_onetrainer_saves_to(tmp_path, monkeypatch):
    saved = []

    class BaseModelSampler:
        @staticmethod
        def save_sampler_output(output, destination, *args, **kwargs):
            saved.append(destination)

    pkg = types.ModuleType("modules")
    sub = types.ModuleType("modules.modelSampler")
    mod = types.ModuleType("modules.modelSampler.BaseModelSampler")
    mod.BaseModelSampler = BaseModelSampler
    monkeypatch.setitem(sys.modules, "modules", pkg)
    monkeypatch.setitem(sys.modules, "modules.modelSampler", sub)
    monkeypatch.setitem(sys.modules, "modules.modelSampler.BaseModelSampler", mod)
    monkeypatch.setattr(sys, "path", list(sys.path))

    assert launcher.install_flat_samples(str(tmp_path)) is True
    dest = str(tmp_path / "samples" / "0 - Ann" / "ts-training-sample-0-0-0")
    BaseModelSampler().save_sampler_output(object(), dest, None, None, None)
    assert saved == [str(tmp_path / "samples" / "ts-training-sample-0-0-0-p0")]
