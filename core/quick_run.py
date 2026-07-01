"""Pure planning for the Home 'Quick Run' unattended pipeline.

The Home Run button drives an unattended chain: optionally detect character names,
optionally caption, then train. This module owns the *decision* of which phases are
needed so it can be unit-tested without Qt; MainWindow executes the phases.
"""

# Phase identifiers, in execution order.
DETECT = "detect"
CAPTION = "caption"
TRAIN = "train"


def plan_phases(subject_type: str, has_roster: bool, is_captioned: bool) -> list:
    """Return the ordered phases a Quick Run should execute.

    - detect: only for Character runs that have no roster yet (filename detection;
      naming is irrelevant for Object/Style, so it is skipped there).
    - caption: only when the dataset is not already captioned (re-captioning stays a
      deliberate Dataset-tab action).
    - train: always the final phase.
    """
    st = (subject_type or "").strip().lower()
    phases = []
    if st in ("character", "person", "face") and not has_roster:
        phases.append(DETECT)
    if not is_captioned:
        phases.append(CAPTION)
    phases.append(TRAIN)
    return phases
