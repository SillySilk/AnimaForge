"""Run the vendored WD14 tagger over a SUBSET of a folder's images.

sd-scripts/finetune/tag_images_by_wd14_tagger.py takes a directory and rewrites
every .tags in it. sd-scripts is fetched at a pinned upstream commit and is not
tracked here, so it must never be patched. Instead: hardlink the to-do images
into a temp directory, tag that, copy the .tags back, discard the temp dir.

Each .tags is copied back the moment the tagger writes it, with a
"[Tagger] (n/N) name: tags" line, so the app can show it on its picture while the
run is still going and the user can stop a run that is tagging badly.

Hardlinks are free on the same NTFS volume; a cross-volume or FAT32 target falls
back to a real copy.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Running as `python scripts/wd14_tag_run.py` puts scripts/ (not the repo root) at
# sys.path[0], so the top-level `core` package would not otherwise be importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


class _TagLander:
    """Copies finished .tags out of the staging dir and announces each one."""

    def __init__(self, stage: Path, dest: Path, images: list):
        self._stage, self._dest = stage, dest
        self._image_by_stem = {Path(p).stem: Path(p).name for p in images}
        self._total = len(images)
        self._sizes = {}
        self._landed = set()

    def sweep(self, settle: bool) -> None:
        """Land every new .tags. With settle, a file must keep the same non-zero size
        across two sweeps first, so a half-written file is never copied."""
        for tag_file in sorted(self._stage.glob("*.tags")):
            name = tag_file.name
            if name in self._landed:
                continue
            try:
                size = tag_file.stat().st_size
            except OSError:
                continue
            if settle and (size == 0 or self._sizes.get(name) != size):
                self._sizes[name] = size
                continue
            shutil.copy2(tag_file, self._dest / name)
            self._landed.add(name)
            tags = tag_file.read_text(encoding="utf-8", errors="replace").strip()
            image = self._image_by_stem.get(tag_file.stem, name)
            print(f"[Tagger] ({len(self._landed)}/{self._total}) {image}: {tags[:120]}",
                  flush=True)


def stage_and_tag(folder: str, only: list, tagger_argv: list, poll: float = 0.5) -> int:
    """Tag exactly `only` (image paths). Returns the tagger's exit code.

    Tags already landed stay in place if the tagger later fails or is stopped,
    the same as any other caption step that is cut short."""
    if not only:
        print("[Tagger] every image already tagged — nothing to do.", flush=True)
        return 0
    src = Path(folder)
    with tempfile.TemporaryDirectory(prefix="af_tag_") as td:
        stage = Path(td)
        for p in only:
            _link_or_copy(Path(p), stage / Path(p).name)
        lander = _TagLander(stage, src, only)
        proc = subprocess.Popen(tagger_argv + [str(stage)])
        while proc.poll() is None:
            lander.sweep(settle=True)
            time.sleep(poll)
        if proc.returncode != 0:
            return proc.returncode
        lander.sweep(settle=False)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image_folder")
    ap.add_argument("--sdscripts", required=True)
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--thresh", type=float, default=0.35)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--onnx", action="store_true")
    ap.add_argument("--force_download", action="store_true")
    ap.add_argument("--only", default="",
                    help="path to a UTF-8 file listing image paths, one per line; "
                         "absent means every image in the folder")
    ap.add_argument("--skip-existing", dest="skip_existing", action="store_true",
                    help="drop images whose .tags sidecar is already non-empty")
    a = ap.parse_args()

    from core.dataset_manager import SUPPORTED_EXTENSIONS
    if a.only:
        only = [ln.strip() for ln in Path(a.only).read_text(encoding="utf-8").splitlines()
                if ln.strip()]
    else:
        only = [str(p) for p in sorted(Path(a.image_folder).iterdir())
                if p.suffix.lower() in SUPPORTED_EXTENSIONS]

    if a.skip_existing:
        def _tagged(p):
            t = Path(p).with_suffix(".tags")
            try:
                return bool(t.read_text(encoding="utf-8").strip())
            except (OSError, UnicodeDecodeError):
                return False
        only = [p for p in only if not _tagged(p)]

    script = str(Path(a.sdscripts) / "finetune" / "tag_images_by_wd14_tagger.py")
    argv = [sys.executable, script,
            f"--repo_id={a.repo_id}", f"--model_dir={a.model_dir}",
            f"--thresh={a.thresh:.2f}", f"--batch_size={a.batch_size}",
            "--caption_extension=.tags", "--remove_underscore"]
    if a.onnx:
        argv.append("--onnx")
    if a.force_download:
        argv.append("--force_download")
    return stage_and_tag(a.image_folder, only, argv)


if __name__ == "__main__":
    raise SystemExit(main())
