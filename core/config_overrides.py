"""Full-control overrides for an engine's generated config.

A run's overrides are a flat {dotted.path: value} dict applied to the final config
dict an engine is about to write. Paths address nested dicts and, by index, lists of
groups (`datasets.0.subsets.0.num_repeats`), so editing one value never freezes its
neighbours. Lists of plain values are leaves.

Some paths are set by the run itself (folders, names, step-derived counts); each
engine lists them as LOCKED_PATHS patterns, and they are never overridden.
"""
import copy
import fnmatch
import json


def _is_group_list(value) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(v, dict) for v in value)


def flatten(config: dict, prefix: str = "") -> dict:
    out = {}
    items = enumerate(config) if isinstance(config, list) else config.items()
    for key, value in items:
        path = f"{prefix}.{key}" if prefix else str(key)
        if (isinstance(value, dict) and value) or _is_group_list(value):
            out.update(flatten(value, path))
        else:
            out[path] = value
    return out


def apply(config: dict, overrides: dict) -> dict:
    out = copy.deepcopy(config)
    for path, value in (overrides or {}).items():
        node = out
        parts = path.split(".")
        for i, part in enumerate(parts):
            last = i == len(parts) - 1
            if isinstance(node, list):
                if not part.isdigit() or int(part) >= len(node):
                    where = ".".join(parts[:i + 1])
                    raise ValueError(f"{path}: {where} is not an entry of that list")
                if last:
                    node[int(part)] = value
                else:
                    node = node[int(part)]
                continue
            if not isinstance(node, dict):
                raise ValueError(f"{path}: {parts[i - 1]!r} is a value, not a group of settings")
            if last:
                node[part] = value
            else:
                node = node.setdefault(part, {})
    return out


def is_locked(path: str, patterns) -> bool:
    return any(fnmatch.fnmatchcase(path, p) for p in patterns or ())


def check_new_path(path: str, flat_base: dict) -> str:
    """Why `path` cannot be added beside `flat_base`'s settings ("" when it can)."""
    parts = (path or "").split(".")
    if not path or any(not p for p in parts):
        return "A setting name can't be empty or have an empty part (like a..b)."
    if path in flat_base:
        return ""
    if any(p.startswith(path + ".") for p in flat_base):
        return f"{path} is a group of settings; add a setting inside it instead."
    for i in range(1, len(parts)):
        if ".".join(parts[:i]) in flat_base:
            return f"{'.'.join(parts[:i])} is a single value; it can't hold other settings."
    return ""


_TRUE, _FALSE = {"true", "yes", "on", "1"}, {"false", "no", "off", "0"}


def parse(text: str, like=None):
    """Parse typed text into the type of `like` (the setting's current value)."""
    raw = (text or "").strip()
    if isinstance(like, bool):
        if raw.lower() in _TRUE:
            return True
        if raw.lower() in _FALSE:
            return False
        raise ValueError(f"expected true or false, got {raw!r}")
    if isinstance(like, int):
        # Whole-number settings like network_alpha also take fractions in the trainers.
        try:
            return int(raw)
        except ValueError:
            pass
        try:
            number = float(raw)
        except ValueError:
            raise ValueError(f"expected a number, got {raw!r}") from None
        return int(number) if number.is_integer() else number
    if isinstance(like, float):
        try:
            return float(raw)
        except ValueError:
            raise ValueError(f"expected a number, got {raw!r}") from None
    if isinstance(like, (list, dict)):
        kind = "list" if isinstance(like, list) else "group"
        try:
            value = json.loads(raw)
        except ValueError:
            raise ValueError(f"expected a JSON {kind}, got {raw!r}") from None
        if not isinstance(value, type(like)):
            raise ValueError(f"expected a JSON {kind}, got {raw!r}")
        return value
    if isinstance(like, str):
        return text
    try:
        return json.loads(raw)
    except ValueError:
        return text


def format_value(value) -> str:
    return value if isinstance(value, str) else json.dumps(value)
