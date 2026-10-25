import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bootstrap import (parse_version, python_ok, pip_commands,
                               is_ssl_cert_error, with_truststore, step_cwd,
                               needs_sd_scripts, sd_scripts_clone_commands,
                               SD_SCRIPTS_COMMIT, torch_step, sd_scripts_update_commands,
                               sd_scripts_state, ensure_sd_scripts)
import pytest
import scripts.bootstrap as bootstrap

PIN = "11d8cedd0c67b93a10f8b2d75f43554f6f12c794"
CU126 = "https://download.pytorch.org/whl/cu126"
CU128 = "https://download.pytorch.org/whl/cu128"


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
    assert any("download.pytorch.org/whl/cu126" in j for j in joined)
    assert any("onnxruntime-gpu" in j for j in joined)
    torch_i = next(i for i, j in enumerate(joined) if "cu126" in j)
    sd_i = next(i for i, j in enumerate(joined)
                if j.replace("\\", "/").endswith("sd-scripts/requirements.txt"))
    assert torch_i < sd_i, "torch (cu126) must install before sd-scripts requirements"


def test_pip_commands_first_is_pip_upgrade():
    cmds = pip_commands("PY", "REPO")
    assert cmds[0] == ["PY", "-m", "pip", "install", "--upgrade", "pip"]


def test_pip_commands_rtx50_uses_cu128_and_torch28():
    joined = [" ".join(c) for c in pip_commands("PY", "REPO", rtx50=True)]
    torch_j = next(j for j in joined if "download.pytorch.org" in j)
    assert "whl/cu128" in torch_j and "torch==2.8.*" in torch_j
    assert not any("cu126" in j for j in joined)


def test_pip_commands_default_is_cu126():
    joined = [" ".join(c) for c in pip_commands("PY", "REPO")]
    torch_j = next(j for j in joined if "download.pytorch.org" in j)
    assert "whl/cu126" in torch_j and "torch==2.8.*" in torch_j


def test_pip_commands_final_step_is_triton():
    cmds = pip_commands("PY", "REPO")
    assert cmds[-1] == ["PY", "-m", "pip", "install", "triton-windows>=3.4,<3.5"]


def test_torch_step_default_is_28_cu126():
    assert torch_step(False) == ["torch==2.8.*", "torchvision", "--index-url", CU126]


def test_torch_step_rtx50_is_28_cu128():
    assert torch_step(True) == ["torch==2.8.*", "torchvision", "--index-url", CU128]


def test_step_cwd_runs_sdscripts_reqs_from_submodule_dir():
    # The sd-scripts requirements file ends with `-e .`; pip resolves `.` against the
    # CWD, so that step MUST run from the sd-scripts dir (where setup.py lives), not
    # the repo root. Other steps use the default cwd (None).
    cmds = pip_commands("PY", "REPO")
    sd_cmd = next(c for c in cmds
                  if " ".join(c).replace("\\", "/").endswith("sd-scripts/requirements.txt"))
    assert step_cwd(sd_cmd, "REPO").replace("\\", "/").endswith("REPO/sd-scripts")
    assert step_cwd(["PY", "-m", "pip", "install", "--upgrade", "pip"], "REPO") is None
    torch_cmd = next(c for c in cmds if "cu126" in " ".join(c))
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


def test_pin_is_v0_12_2():
    assert SD_SCRIPTS_COMMIT == PIN


def test_sd_scripts_update_commands_shape():
    dest = str(Path("R") / "sd-scripts")
    assert sd_scripts_update_commands(dest, "abc") == [
        ["git", "-C", dest, "fetch", "origin"],
        ["git", "-C", dest, "checkout", "abc"],
    ]


def _fake_run(head="", porcelain="", toplevel=None, rc=0):
    class R:
        def __init__(self, out):
            self.returncode, self.stdout, self.stderr = rc, out, "boom"

    def run(cmd, **kw):
        if "--show-toplevel" in cmd:
            return R(toplevel if toplevel is not None else cmd[2])
        assert "--untracked-files=no" in cmd or "status" not in cmd
        return R(head if "rev-parse" in cmd else porcelain)
    return run


def _repo_with_sd(tmp_path):
    d = tmp_path / "sd-scripts"
    d.mkdir(exist_ok=True)
    return str(d)


def _repo_with_sd_named(tmp_path, name):
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    return str(d)


def test_sd_scripts_state_absent(tmp_path):
    assert sd_scripts_state(str(tmp_path / "sd-scripts"), run=_fake_run()) == "absent"


def test_sd_scripts_state_at_pin(tmp_path):
    run = _fake_run(head=PIN + "\n", porcelain="")
    assert sd_scripts_state(_repo_with_sd(tmp_path), run=run) == "at_pin"


def test_sd_scripts_state_behind(tmp_path):
    run = _fake_run(head="1a3ec9ea745fe9883551dfca5c947ea3d6aa68c7\n", porcelain="")
    assert sd_scripts_state(_repo_with_sd(tmp_path), run=run) == "behind"


def test_sd_scripts_state_dirty(tmp_path):
    run = _fake_run(head=PIN + "\n", porcelain=" M train_network.py\n")
    assert sd_scripts_state(_repo_with_sd(tmp_path), run=run) == "dirty"


def test_sd_scripts_state_untracked_wd14_models_not_dirty(tmp_path):
    run = _fake_run(head=PIN + "\n", porcelain="")  # --untracked-files=no hides ?? entries
    assert sd_scripts_state(_repo_with_sd(tmp_path), run=run) == "at_pin"
    seen = []

    def spy(cmd, **kw):
        seen.append(cmd)
        return run(cmd, **kw)
    sd_scripts_state(_repo_with_sd_named(tmp_path, "x"), run=spy)
    status = [c for c in seen if "status" in c][0]
    assert "--untracked-files=no" in status


def test_sd_scripts_state_git_failure_is_clear_error(tmp_path):
    with pytest.raises(SystemExit) as e:
        sd_scripts_state(_repo_with_sd(tmp_path), run=_fake_run(rc=128))
    assert "failed" in str(e.value) and "re-run install.bat" in str(e.value)


def test_sd_scripts_state_toplevel_mismatch_is_clear_error(tmp_path):
    run = _fake_run(head=PIN + "\n", toplevel=str(tmp_path))
    with pytest.raises(SystemExit) as e:
        sd_scripts_state(_repo_with_sd(tmp_path), run=run)
    assert "not its own git checkout" in str(e.value)
    assert "Delete or rename" in str(e.value)


def test_ensure_sd_scripts_refuses_dirty(monkeypatch):
    monkeypatch.setattr(bootstrap, "sd_scripts_state", lambda repo, run=None: "dirty")
    with pytest.raises(SystemExit) as e:
        ensure_sd_scripts()
    assert "sd-scripts has local changes" in str(e.value)


def test_ensure_sd_scripts_at_pin_is_noop(monkeypatch):
    monkeypatch.setattr(bootstrap, "sd_scripts_state", lambda repo, run=None: "at_pin")
    monkeypatch.setattr(bootstrap, "_run", lambda *a, **k: pytest.fail("ran a command"))
    ensure_sd_scripts()


def test_ensure_sd_scripts_behind_updates(monkeypatch):
    calls = []

    class R:
        returncode = 0
    monkeypatch.setattr(bootstrap, "sd_scripts_state", lambda repo, run=None: "behind")
    monkeypatch.setattr(bootstrap, "_run", lambda cmd, **k: calls.append(cmd) or R())
    ensure_sd_scripts()
    assert [c[3] for c in calls] == ["fetch", "checkout"]
    assert calls[1][-1] == PIN
