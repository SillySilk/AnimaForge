"""The final previews: after the trainer has saved and the file has passed every
check, the shipped LoRA is loaded back and its preview prompts rendered, so the
previews always show the file that ships.

The trainer's own previews stop at the last sampling interval. Hailey
(2026-09-04) sampled at step 989 of 1139 and looked perfect there; the file it
shipped was the state at 1139, which rendered a mosaic. Nothing had drawn it.
"""
import py_compile
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.engines.onetrainer import render
from core.engines.onetrainer.engine import TrainerInstall


def _install(tmp_path):
    return TrainerInstall(root=str(tmp_path), python=str(tmp_path / "py.exe"),
                          train_script=str(tmp_path / "train.py"))


def test_the_render_runs_the_apps_script_under_onetrainers_python(tmp_path):
    """The script lives in the app but runs in OneTrainer's interpreter, from
    OneTrainer's root -- the same arrangement as train.py."""
    cmd = render.command(_install(tmp_path), r"C:\run\config.json",
                         r"C:\run\Aria.safetensors", r"C:\run\workspace\samples\final")
    assert cmd[0] == str(tmp_path / "py.exe")
    assert cmd[1] == render.SCRIPT
    assert cmd[2:] == ["--config-path", r"C:\run\config.json",
                       "--lora", r"C:\run\Aria.safetensors",
                       "--destination", r"C:\run\workspace\samples\final"]


def test_the_script_exists_and_compiles():
    """It is never imported by the app, so nothing else would notice a typo
    until the end of a real run."""
    assert Path(render.SCRIPT).is_file()
    py_compile.compile(render.SCRIPT, doraise=True)


def test_the_script_imports_nothing_from_the_app():
    """It runs where OneTrainer's modules are importable and the app's are not."""
    text = Path(render.SCRIPT).read_text(encoding="utf-8")
    assert "from core" not in text and "import core" not in text
    assert "from ui" not in text and "from utils" not in text


def test_final_previews_land_beside_the_training_previews(tmp_path):
    run = tmp_path / "Aria"
    assert Path(render.destination(str(run))) == run / "workspace" / "samples" / "final"


def test_count_images_reads_what_the_render_wrote(tmp_path):
    dest = tmp_path / "final"
    assert render.count_images(str(dest)) == 0
    (dest / "0 - final").mkdir(parents=True)
    (dest / "0 - final" / "a.jpg").write_bytes(b"x")
    (dest / "1 - final").mkdir()
    (dest / "1 - final" / "b.png").write_bytes(b"x")
    (dest / "notes.txt").write_text("x", encoding="utf-8")
    assert render.count_images(str(dest)) == 2
