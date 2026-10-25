"""Which accelerate launcher path a run takes.

A user hit "use_libuv was requested but PyTorch was built without libuv support"
*while the log showed USE_LIBUV=0 had been set*. Reproduced locally: the cause is
not the environment variable at all.

`accelerate launch` loads ~/.cache/huggingface/accelerate/default_config.yaml
whenever that file exists (launch.py: `if args.config_file is not None or
os.path.isfile(default_config_file)`), and a `distributed_type: MULTI_GPU` in it
sets args.multi_gpu=True *even though we pass --num_processes=1*. That routes the
launch through multi_gpu_launcher -> torch elastic -> static_tcp_rendezvous, which
constructs TCPStore() with no use_libuv kwarg — so it never consults USE_LIBUV and
the env var cannot help. torch only reads USE_LIBUV in distributed/rendezvous.py
and elastic/utils/distributed.py, neither of which is on that path.

Anyone who ever ran `accelerate config` (every Kohya/sd-scripts guide says to) has
such a file, and AnimaForge inherits it. Passing our own single-process config
pins simple_launcher, which builds no TCPStore at all.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QCoreApplication

QCoreApplication.instance() or QCoreApplication([])

from core.trainer import ACCELERATE_CONFIG_NAME, TrainingProcess, write_accelerate_config


def test_our_accelerate_config_pins_the_single_process_launcher(tmp_path):
    """distributed_type NO is what keeps accelerate on simple_launcher."""
    path = write_accelerate_config(tmp_path)
    body = path.read_text(encoding="utf-8")
    assert "distributed_type: 'NO'" in body
    assert "num_processes: 1" in body
    assert "num_machines: 1" in body


def test_the_config_is_written_where_the_run_can_find_it(tmp_path):
    path = write_accelerate_config(tmp_path)
    assert path == tmp_path / ACCELERATE_CONFIG_NAME
    assert path.is_file()


def test_writing_the_config_creates_a_missing_directory(tmp_path):
    """Never fail a run over a directory we can create ourselves."""
    path = write_accelerate_config(tmp_path / "configs")
    assert path.is_file()


def test_build_command_pins_accelerate_to_our_config(tmp_path):
    """Without this, a stray MULTI_GPU default_config.yaml picks the launcher."""
    config = tmp_path / "Horchata_config.toml"
    config.write_text("", encoding="utf-8")
    _, args = TrainingProcess().build_command(str(config), str(tmp_path))

    accel = tmp_path / ACCELERATE_CONFIG_NAME
    assert f"--config_file={accel}" in args
    assert accel.is_file()


def test_the_accelerate_config_precedes_the_training_script(tmp_path):
    """After the script path it would be passed *to* sd-scripts, not to accelerate."""
    config = tmp_path / "Horchata_config.toml"
    config.write_text("", encoding="utf-8")
    _, args = TrainingProcess().build_command(str(config), str(tmp_path))

    script = next(i for i, a in enumerate(args) if a.endswith("anima_train_network.py"))
    accel = next(i for i, a in enumerate(args) if ACCELERATE_CONFIG_NAME in a)
    assert accel < script


def test_the_training_config_still_reaches_sd_scripts(tmp_path):
    """Two different --config_file arguments; the run dies if we conflate them."""
    config = tmp_path / "Horchata_config.toml"
    config.write_text("", encoding="utf-8")
    _, args = TrainingProcess().build_command(str(config), str(tmp_path))

    script = next(i for i, a in enumerate(args) if a.endswith("anima_train_network.py"))
    assert f"--config_file={config}" in args[script + 1:]


def test_build_command_moss_uses_moss_script(tmp_path):
    config = tmp_path / "x_config.toml"
    config.write_text("", encoding="utf-8")
    _, args = TrainingProcess().build_command(str(config), str(tmp_path), model="moss")

    assert sum(a.endswith("moss_image_train_network.py") for a in args) == 1
    assert not any(a.endswith("anima_train_network.py") for a in args)
    assert "--mixed_precision=bf16" in args


def test_build_command_anima_unchanged(tmp_path):
    config = tmp_path / "x_config.toml"
    config.write_text("", encoding="utf-8")
    proc = TrainingProcess()
    program, args = proc.build_command(str(config), str(tmp_path))
    accel = tmp_path / ACCELERATE_CONFIG_NAME
    assert args == [
        "-m", "accelerate.commands.launch",
        f"--config_file={accel}",
        "--num_cpu_threads_per_process=2",
        "--num_processes=1",
        "--num_machines=1",
        "--mixed_precision=bf16",
        "--dynamo_backend=no",
        str(tmp_path / "anima_train_network.py"),
        f"--config_file={config}",
    ]
    assert (program, args) == proc.build_command(str(config), str(tmp_path), model="anima")
