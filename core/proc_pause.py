"""Pause / resume a whole training process tree.

accelerate and OneTrainer both launch the real trainer as a child, so the root and
every descendant are suspended together — pausing only the launcher leaves the GPU
work running. Suspended processes keep their VRAM; resuming continues the same step.
"""
import psutil


def set_tree_suspended(pid: int, suspended: bool) -> bool:
    """Suspend (or resume) `pid` and all its descendants. Returns whether the root
    process was found; a child that exits mid-walk is skipped."""
    if not pid:
        return False
    try:
        root = psutil.Process(int(pid))
        tree = [root] + root.children(recursive=True)
    except psutil.Error:
        return False
    # Children first on suspend so none races ahead of a stopped parent; parent first
    # on resume so it is ready to read its children's output.
    order = list(reversed(tree)) if suspended else tree
    for proc in order:
        try:
            proc.suspend() if suspended else proc.resume()
        except psutil.Error:
            continue
    return True
