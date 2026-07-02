import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bootstrap import (parse_version, python_ok, pip_commands,
                               is_ssl_cert_error, with_truststore, step_cwd,
                               needs_sd_scripts, sd_scripts_clone_commands,
                               SD_SCRIPTS_COMMIT)


def test_parse_version():
    assert parse_version("Python 3.10.14") == (3, 10, 14)
    assert parse_version("Python 3.11.9") == (3, 11, 9)
    assert parse_version("garbage") is None
    assert parse_version("") is None


def test_python_ok_supported():
    assert python_ok("Python 3.10.14")
    assert python_ok("Python 3.11.9")


def test_python_ok_rejects_out_of_range():
    assert not python_ok("Python 3.9.18")
    assert not python_ok("Python 3.12.0")
    assert not python_ok("Python 2.7.18")
    assert not python_ok("nonsense")


def test_pip_commands_torch_before_sdscripts():
    cmds = pip_commands("PY", "REPO")
    joined = [" ".join(c) for c in cmds]
    assert any("download.pytorch.org/whl/cu121" in j for j in joined)
    assert any("onnxruntime-gpu" in j for j in joined)
    torch_i = next(i for i, j in enumerate(joined) if "cu121" in j)
    sd_i = next(i for i, j in enumerate(joined)
                if j.replace("\\", "/").endswith("sd-scripts/requirements.txt"))
    assert torch_i < sd_i, "torch (cu121) must install before sd-scripts requirements"


def test_pip_commands_first_is_pip_upgrade():
    cmds = pip_commands("PY", "REPO")
    assert cmds[0] == ["PY", "-m", "pip", "install", "--upgrade", "pip"]


def test_pip_commands_rtx50_uses_cu128_and_torch27():
    joined = [" ".join(c) for c in pip_commands("PY", "REPO", rtx50=True)]
    torch_j = next(j for j in joined if "download.pytorch.org" in j)
    assert "whl/cu128" in torch_j and "torch>=2.7" in torch_j
    assert not any("cu121" in j for j in joined)


def test_pip_commands_default_stays_cu121():
    joined = [" ".join(c) for c in pip_commands("PY", "REPO")]
    torch_j = next(j for j in joined if "download.pytorch.org" in j)
    assert "whl/cu121" in torch_j and "torch>=2.5" in torch_j


def test_step_cwd_runs_sdscripts_reqs_from_submodule_dir():
    # The sd-scripts requirements file ends with `-e .`; pip resolves `.` against the
    # CWD, so that step MUST run from the sd-scripts dir (where setup.py lives), not
    # the repo root. Other steps use the default cwd (None).
    cmds = pip_commands("PY", "REPO")
    sd_cmd = next(c for c in cmds
                  if " ".join(c).replace("\\", "/").endswith("sd-scripts/requirements.txt"))
    assert step_cwd(sd_cmd, "REPO").replace("\\", "/").endswith("REPO/sd-scripts")
    assert step_cwd(["PY", "-m", "pip", "install", "--upgrade", "pip"], "REPO") is None
    torch_cmd = next(c for c in cmds if "cu121" in " ".join(c))
    assert step_cwd(torch_cmd, "REPO") is None


def test_is_ssl_cert_error():
    assert is_ssl_cert_error("... [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed ...")
    assert not is_ssl_cert_error("ERROR: Could not find a version that satisfies torch")
    assert not is_ssl_cert_error("")


def test_with_truststore_inserts_flag_after_install():
    cmd = ["PY", "-m", "pip", "install", "torch", "--index-url", "URL"]
    out = with_truststore(cmd)
    assert out == ["PY", "-m", "pip", "install", "--use-feature=truststore",
                   "torch", "--index-url", "URL"]


def test_with_truststore_idempotent_and_noop_without_install():
    already = ["PY", "-m", "pip", "install", "--use-feature=truststore", "torch"]
    assert with_truststore(already) == already
    no_install = ["git", "submodule", "update"]
    assert with_truststore(no_install) == no_install


def test_needs_sd_scripts_true_when_absent(tmp_path):
    assert needs_sd_scripts(str(tmp_path)) is True


def test_needs_sd_scripts_false_when_present(tmp_path):
    d = tmp_path / "sd-scripts"
    d.mkdir()
    (d / "anima_train_network.py").write_text("# anima\n", encoding="utf-8")
    assert needs_sd_scripts(str(tmp_path)) is False


def test_sd_scripts_clone_commands_shape(tmp_path):
    cmds = sd_scripts_clone_commands(str(tmp_path), "abc123")
    assert cmds[0][:2] == ["git", "clone"]
    assert "https://github.com/kohya-ss/sd-scripts.git" in cmds[0]
    assert cmds[1] == ["git", "-C", str(tmp_path / "sd-scripts"), "checkout", "abc123"]


def test_pin_is_the_anima_commit():
    assert SD_SCRIPTS_COMMIT == "1a3ec9ea745fe9883551dfca5c947ea3d6aa68c7"
