"""Minimal pytest-free runner (network-restricted env has no pytest).

Discovers test_* functions in the test modules and runs them, supplying a
fresh temp directory Path for any function that declares a `tmp_path` param.
"""
import sys
import inspect
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import test_config_generator as m1
import test_dataset_manager as m2
import test_state_utils as m3
import test_batch as m4
import test_llm_refine as m5
import test_settings as m6
import test_forge_api as m7
import test_characters as m8
import test_train_log as m9
import test_sample_prompts as m10
import test_sets as m11
import test_gpu_check as m12
import test_train_tab_sets as m13
import test_gen_assets as m14
import test_styles as m15
import test_home_tab as m16
import test_main_window_shell as m17
import test_image_editor as m18
import test_characters_tab as m20
import test_dataset_workflow as m21
import test_paths as m23
import test_env as m24
import test_bootstrap as m25
import test_model_locations as m26
import test_lowvram as m27
import test_workflow as m28
import test_step_calculator as m29
import test_quick_run as m30
import test_naming as m31
import test_name_validate as m32
import test_home_run_split as m33
import test_caption_progress as m34
import test_dataset_tab_captioning as m35
import test_batch_status as m36
import test_headless as m37
import test_caption_policy as m38
import test_caption_manifest as m39
import test_caption_runner as m40


class MonkeyPatch:
    """Just enough of pytest's monkeypatch fixture: setattr with undo."""

    def __init__(self):
        self._undo = []

    def setattr(self, target, name, value):
        self._undo.append((target, name, getattr(target, name)))
        setattr(target, name, value)

    def undo(self):
        for target, name, old in reversed(self._undo):
            setattr(target, name, old)
        self._undo.clear()


def run_module(mod):
    passed = failed = 0
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if not name.startswith("test_"):
            continue
        if fn.__module__ != mod.__name__:
            continue
        params = inspect.signature(fn).parameters
        kwargs = {}
        mp = None
        if "monkeypatch" in params:
            mp = kwargs["monkeypatch"] = MonkeyPatch()
        try:
            if "tmp_path" in params:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d), **kwargs)
            else:
                fn(**kwargs)
            print(f"  PASS {mod.__name__}.{name}")
            passed += 1
        except Exception:
            print(f"  FAIL {mod.__name__}.{name}")
            traceback.print_exc()
            failed += 1
        finally:
            if mp is not None:
                mp.undo()
    return passed, failed


if __name__ == "__main__":
    total_p = total_f = 0
    for mod in (m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, m16, m17, m18, m20, m21, m23, m24, m25, m26, m27, m28, m29, m30, m31, m32, m33, m34, m35, m36, m37, m38, m39, m40):
        print(f"== {mod.__name__} ==")
        p, f = run_module(mod)
        total_p += p
        total_f += f
    print(f"\nTOTAL: {total_p} passed, {total_f} failed")
    sys.exit(1 if total_f else 0)
