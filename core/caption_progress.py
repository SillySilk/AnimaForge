"""Parse per-image progress out of the captioning workers' log_line stream.

The slow steps print lines like:
  [JoyCaption] (3/12) aria_03.png: <caption>…
  [LLM] (5/12) my photo.png: <prose>… | tags: …
This turns one such line into a ProgressTick; anything else returns None.
"""
import re
from dataclasses import dataclass

_PHASES = {"tagger": "Tag", "joycaption": "Describe", "llm": "Refine"}
_PREFIX_RE = re.compile(r"^\[(\w+)\]")
# (n/N) then the filename up to the first colon (Windows filenames can't contain ':')
_PROG_RE = re.compile(r"\((\d+)\s*/\s*(\d+)\)\s+(.+?):")


@dataclass
class ProgressTick:
    phase: str
    done: int
    total: int
    filename: str


def parse_progress(line):
    """Return a ProgressTick for a per-image worker line, else None. Never raises."""
    if not line:
        return None
    prefix = _PREFIX_RE.match(line.strip())
    if not prefix:
        return None
    phase = _PHASES.get(prefix.group(1).lower())
    if phase is None:
        return None
    prog = _PROG_RE.search(line)
    if not prog:
        return None
    return ProgressTick(phase=phase, done=int(prog.group(1)),
                        total=int(prog.group(2)), filename=prog.group(3).strip())
