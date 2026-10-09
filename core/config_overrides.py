"""Full-control overrides for an engine's generated config.

A run's overrides are a flat {dotted.path: value} dict applied to the final config
dict an engine is about to write. Paths address nested dicts; lists are leaves.
"""
import copy
import json


def flatten(config: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in config.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict) and value:
            out.update(flatten(value, path))
        else:
            out[path] = value
    return out


def apply(config: dict, overrides: dict) -> dict:
    out = copy.deepcopy(config)
    for path, value in (overrides or {}).items():
        node = out
        parts = path.split(".")
        for part in parts[:-1]:
            nxt = node.setdefault(part, {})
            if not isinstance(nxt, dict):
                raise ValueError(f"{path}: {part!r} is a value, not a group of settings")
            node = nxt
        node[parts[-1]] = value
    return out


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
        try:
            return int(raw)
        except ValueError:
            raise ValueError(f"expected a whole number, got {raw!r}") from None
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
