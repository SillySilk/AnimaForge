from core import joycaption


def test_name_flag_is_passed_when_set():
    args = joycaption.build_args("run.py", "C:/ds", name="mrx")
    assert "--name=mrx" in args


def test_no_name_flag_when_blank():
    assert not any(a.startswith("--name") for a in joycaption.build_args("run.py", "C:/ds", name="  "))


def test_existing_flags_unchanged():
    args = joycaption.build_args("run.py", "C:/ds", overwrite=True, model_id="m/x", cache_dir="D:/c")
    assert args[:3] == ["run.py", "C:/ds", "--ext=.nl"]
    assert "--overwrite" in args and "--model_id=m/x" in args and "--cache_dir=D:/c" in args
