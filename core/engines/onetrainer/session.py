"""One training run, start to finish: preflight → generate → training → outcome.

The session owns the run folder (a complete record of how the LoRA was made:
config, concepts, samples, trainer.log, workspace, cache, final .safetensors)
and translates TrainerProcess events into UI-facing signals. One run at a time.
"""
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from core import paths, sysmem
from core.engines.onetrainer import concepts, engine, metadata, preset, render, snapshots, steps, verify
from core.engines.onetrainer.process import TrainerProcess

_TAIL_LINES = 15
_OOM_HINT = ("CUDA ran out of memory. Close anything else using the GPU "
             "(games, browsers, other AI apps) and press Train again.")

# 0xC0000005, as QProcess reports it. Windows kills a process that dereferences a
# failed allocation with this, and that is precisely how running out of system
# commit charge presents: PyTorch asks for memory mid-quantization, gets null,
# c10.dll faults. Nothing is printed -- no traceback, no CUDA OOM marker for
# _OOM_MARKERS to catch -- so trainer.log just stops and the exit code is the
# only evidence left. Preflight refuses runs that start short, but a run can also
# be crowded out by something launched after it began.
ACCESS_VIOLATION = -1073741819
_CRASH_HINT = (
    "Windows killed the trainer while it was still loading the model. That is "
    "what running out of system memory looks like here — RAM plus pagefile, not "
    "VRAM, so it happens with a nearly idle GPU and no error message. Close other "
    "AI apps and press Train again.")


@dataclass
class RunParams:
    lora_name: str
    dataset_dir: str
    image_count: int
    concept: str
    trigger_word: str
    output_base: str
    onetrainer_root: str
    # Prepended to every caption at Build, so it is trained scaffolding and belongs
    # in the finished file's trained words alongside the trigger.
    quality_prefix: str = ""
    # Setup's "Target steps"; 0 means Auto (the per-concept steps-per-image rate).
    target_steps: int = 0
    # Setup's "Write profiler traces": OneTrainer's OT_DEBUG_PROFILES switch. The
    # traces land in the trainer's own folder and are moved into the run folder.
    debug_profiles: bool = False


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=4), encoding="utf-8")


class TrainingSession(QObject):
    IDLE, PREFLIGHT, GENERATING, TRAINING = "idle", "preflight", "generating", "training"
    # After the trainer has exited and the file has passed every check: one more
    # OneTrainer pass renders the shipped file's preview prompts (render.py).
    RENDERING = "rendering"
    DONE, FAILED, CANCELED = "done", "failed", "canceled"

    state_changed = Signal(str)
    progress = Signal(dict)
    failed = Signal(str)
    done = Signal(str)          # absolute path of the finished .safetensors
    # The third ending. It used to be state_changed alone, which nothing in the UI
    # listens to: a cancelled run left Home on "Starting trainer…" with Cancel lit
    # and Train grey until the app was restarted -- a dead run that looked alive
    # (Connie_D'Amico, 2026-09-04).
    canceled = Signal()

    def __init__(self, parent=None, process_factory=TrainerProcess):
        super().__init__(parent)
        self._factory = process_factory
        self._state = self.IDLE
        self._proc = None
        self._run_dir = None
        self._output_path = ""
        self._log = None
        self._tail = []
        self._canceled = False
        self._epoch = {"n": 0, "total": 0}
        self._step = {"n": 0, "total": 0}
        self._plan = None
        self._params = None
        self._install = None
        self._spawn_stamp = 0.0
        self._notes = []
        self._watch = verify.DivergenceWatch()
        self._diverged = None

    # ---- public API ----

    def is_running(self) -> bool:
        return self._state in (self.PREFLIGHT, self.GENERATING, self.TRAINING,
                               self.RENDERING)

    def run_dir(self) -> str:
        return str(self._run_dir) if self._run_dir else ""

    def samples_dir(self) -> str:
        return str(self._run_dir / "workspace" / "samples") if self._run_dir else ""

    def notes(self) -> list:
        """What the user should know about a run that finished DONE anyway --
        today, that the end of the run overshot and a snapshot shipped instead.
        Each entry is a finished sentence; empty for an uneventful run."""
        return list(self._notes)

    def start(self, p: RunParams) -> bool:
        if self.is_running():
            self.failed.emit("A training run is already in progress.")
            return False
        self._canceled = False
        self._params = p
        self._tail = []
        self._epoch = {"n": 0, "total": 0}
        self._step = {"n": 0, "total": 0}
        self._run_dir = None
        self._output_path = ""
        self._notes = []
        self._watch = verify.DivergenceWatch()
        self._diverged = None
        self._set_state(self.PREFLIGHT)
        # Settings store install-internal paths portable (relative to the app root).
        # The trainer subprocess runs with cwd = the OneTrainer root, so a relative
        # base has to be resolved HERE — before preflight measures the disk and
        # before a single path is baked into the run config.
        output_base = paths.from_portable(p.output_base)
        problems = engine.preflight(p.onetrainer_root, p.dataset_dir,
                                    p.image_count, p.trigger_word, output_base)
        if problems:
            self._fail("\n".join(problems))
            return False
        install = engine.locate(p.onetrainer_root)
        self._install = install

        self._set_state(self.GENERATING)
        try:
            template = preset.load_template()
        except preset.TemplateError as e:
            self._fail(str(e))
            return False
        safe_name = paths.sanitize_name(p.lora_name)
        run_dir = Path(paths.run_output_dir(output_base, p.lora_name))
        run_dir.mkdir(parents=True, exist_ok=True)
        self._run_dir = run_dir
        self._plan = steps.plan(p.image_count, p.concept, p.target_steps)
        cfg = preset.build_run_config(template, str(run_dir), safe_name,
                                      self._plan["epochs"], self._plan["warmup_steps"],
                                      self._plan["prodigy_steps"])
        self._output_path = cfg["output_model_destination"]
        try:
            _write_json(run_dir / "concepts.json", concepts.build_concepts(p.dataset_dir))
            _write_json(run_dir / "samples.json",
                        concepts.build_samples(p.dataset_dir, p.trigger_word))
            _write_json(run_dir / "config.json", cfg)
            self._log = open(run_dir / "trainer.log", "w", encoding="utf-8", errors="replace")
        except OSError as e:
            self._fail(f"Could not prepare the run folder: {e}")
            return False

        proc = self._factory(self)
        proc.line.connect(self._on_line)
        proc.progress.connect(self._on_progress)
        proc.finished.connect(self._on_finished)
        self._proc = proc
        self._set_state(self.TRAINING)
        self._spawn_stamp = time.time()
        extra_env = {"OT_DEBUG_PROFILES": "1"} if p.debug_profiles else None
        if not proc.start(install, str(run_dir / "config.json"), extra_env=extra_env):
            self._close_log()
            self._fail("Could not start the OneTrainer process.")
            return False
        return True

    def cancel(self):
        if not self.is_running():
            return
        self._canceled = True
        if self._proc is not None:
            self._proc.cancel()

    # ---- internal ----

    def _set_state(self, state: str):
        self._state = state
        self.state_changed.emit(state)

    def _fail(self, message: str):
        self._set_state(self.FAILED)
        self.failed.emit(message)

    def _note(self, text: str):
        """Append a line to the run's trainer.log after the trainer has exited.

        The run folder is the record of how the LoRA was made, so anything this
        code does to the finished file belongs in it — including failing to.
        """
        if self._run_dir is None:
            return
        try:
            with open(self._run_dir / "trainer.log", "a", encoding="utf-8",
                      errors="replace") as f:
                print("[animaforge] " + text, file=f)
        except OSError:
            pass

    def _stamp_trained_words(self):
        """Write the trigger and the set's recurring words into the finished LoRA.

        OneTrainer leaves the file with no metadata at all, so nothing downstream
        can say what word calls this LoRA — see core/engines/onetrainer/metadata.py. Runs after
        verify, on a file that is already known good, and never fails the run: the
        LoRA is finished and usable whether or not its label got attached, so a
        failure here is a line in trainer.log, not an hour thrown away.
        """
        p = self._params
        if p is None:
            return
        try:
            written = metadata.stamp(
                self._output_path, p.lora_name, p.dataset_dir, p.trigger_word,
                p.quality_prefix, self._plan, p.image_count)
        except (metadata.MetadataError, OSError, ValueError) as e:
            self._note(f"trained words were not written into the LoRA: {e}")
            return
        self._note("trained words written into the LoRA and to "
                   + Path(written["sidecar"]).name + ": "
                   + ", ".join(written["words"]))

    def _ship_snapshot(self, found: dict) -> bool:
        """The run overshot or blew up: ship the last snapshot from before the
        loss turned.

        Returns False when no such snapshot exists (or none of them is a usable
        LoRA), leaving the overshot file exactly where the trainer put it.
        """
        save_dir = self._run_dir / "workspace" / "save"
        for snap in snapshots.candidates(snapshots.find(str(save_dir)),
                                         found["last_good_epoch"]):
            if verify.verify_output(snap.path) is not None:
                self._note(f"snapshot from epoch {snap.epoch} skipped: not a usable LoRA")
                continue
            try:
                kept = snapshots.promote(snap.path, self._output_path)
            except OSError as e:
                self._note(f"snapshot from epoch {snap.epoch} could not be shipped: {e}")
                return False
            note = (verify.describe(found)
                    + f" Shipped the snapshot from epoch {snap.epoch} (step {snap.step}) "
                    f"instead, the last one from before the loss turned.")
            if Path(kept).is_file():
                note += f" The broken end is kept beside it as {Path(kept).name}."
            self._notes.append(note)
            self._note(note)
            return True
        return False

    def _discard_snapshots(self):
        """The question the snapshots existed to answer is settled; ~114 MB each
        is not worth keeping. The run folder stays the record of the run without
        them: config, log, samples, and the file(s) that matter."""
        removed = snapshots.discard(str(self._run_dir / "workspace" / "save"))
        if removed:
            self._note(f"{removed} snapshot(s) removed from workspace\\save")

    def _finish_diverged(self, found: dict):
        """The trainer was stopped on a blow-up: ship the last snapshot from before
        it, or fail saying why nothing could be shipped."""
        if self._ship_snapshot(found):
            self._discard_snapshots()
            self._stamp_trained_words()
            self._start_render()
            return
        self._discard_snapshots()
        self._fail(
            f"{verify.describe(found)}\n\nNo usable snapshot from epoch "
            f"{found['last_good_epoch']} or earlier was found in the run's "
            "workspace\\save folder, so there is no LoRA to ship. Train again.")

    def _close_log(self):
        if self._log is not None:
            self._log.close()
            self._log = None

    # ---- the final previews ----

    def _start_render(self):
        """Load the shipped file back and render its preview prompts, so the
        previews show the file that ships and not the last sampling interval.
        A render that cannot start is a note; the run lands as DONE either way."""
        dest = render.destination(str(self._run_dir))
        try:
            self._log = open(self._run_dir / "trainer.log", "a", encoding="utf-8",
                             errors="replace")
        except OSError:
            self._log = None
        proc = self._factory(self)
        proc.line.connect(self._on_line)
        proc.finished.connect(self._on_finished)
        self._proc = proc
        self._set_state(self.RENDERING)
        started = proc.start_script(
            self._install, render.SCRIPT,
            render.args(str(self._run_dir / "config.json"), self._output_path, dest))
        if not started:
            self._proc = None
            self._close_log()
            self._preview_note("The final previews were not rendered: the render "
                               "process could not be started. The LoRA itself is fine.")
            self._land_done()

    def _preview_note(self, text: str):
        self._notes.append(text)
        self._note(text)

    def _land_done(self):
        self._set_state(self.DONE)
        self.done.emit(self._output_path)

    def _on_render_finished(self, exit_code: int):
        self._close_log()
        self._proc = None
        dest = render.destination(str(self._run_dir))
        written = render.count_images(dest)
        if self._canceled:
            self._preview_note("The final previews were skipped: the render was "
                               "cancelled. The LoRA itself is finished and verified.")
        elif exit_code == 0 and written:
            self._note(f"{written} final preview(s) of the shipped file rendered into "
                       f"workspace\\samples\\final")
        else:
            self._preview_note(
                f"The final previews of the finished LoRA did not render (exit code "
                f"{exit_code}, {written} image(s) written). The LoRA itself is fine; "
                f"the render's output is at the end of trainer.log.")
        self._land_done()

    def _on_line(self, text: str):
        if self._log is not None:
            self._log.write(text + "\n")
        self._tail.append(text)
        del self._tail[:-_TAIL_LINES]
        if self._state == self.TRAINING and self._diverged is None:
            found = self._watch.feed(text)
            if found is not None:
                self._stop_diverged(found)

    def _stop_diverged(self, found: dict):
        """The loss just blew up. Every epoch from here trains noise, and runs are
        left unattended -- so stop the trainer now; _on_finished ships the last
        snapshot from before the jump."""
        self._diverged = found
        if self._log is not None:
            self._log.write("[animaforge] " + verify.describe(found)
                            + " Stopping the trainer.\n")
            self._log.flush()
        if self._proc is not None:
            self._proc.cancel()

    def _on_progress(self, d: dict):
        target = self._epoch if d["name"] == "epoch" else self._step
        target["n"], target["total"] = d["n"], d["total"]
        # The plan knows the run's real size before the trainer's first `step:` bar
        # arrives; without it the opening stretch reports a total of 0 and the band
        # sits at 0/0 for minutes.
        # The trainer's own bars are ground truth once they arrive; the plan fills in
        # until then. Without it the opening stretch reports a total of 0 and the band
        # sits at 0/0 for minutes.
        plan = self._plan or {}
        step_total = self._step["total"] or plan.get("steps_per_epoch", 0)
        epoch_total = self._epoch["total"] or plan.get("epochs", 0)
        payload = {
            "epoch": self._epoch["n"], "epoch_total": self._epoch["total"],
            "step": self._step["n"], "step_total": step_total,
            "loss": d.get("loss"),
            "global_step": self._epoch["n"] * step_total + self._step["n"],
            "global_total": epoch_total * step_total or plan.get("total_steps", 0),
        }
        self.progress.emit(payload)

    # OneTrainer writes its profiler traces relative to its working directory --
    # the install root, not the run -- under these names (GenericTrainer, #1682).
    PROFILE_GLOBS = ("profile-step*.json", "memory-step*.pickle")

    def _collect_profiles(self):
        """Move this run's profiler traces from the trainer's folder into the run
        folder, where the rest of the record is. Only files younger than the spawn
        are this run's; an older one belongs to a run that was never collected."""
        if self._run_dir is None or self._install is None:
            return
        for pattern in self.PROFILE_GLOBS:
            for path in Path(self._install.root).glob(pattern):
                try:
                    if path.stat().st_mtime < self._spawn_stamp - 2:
                        continue
                    shutil.move(str(path), str(self._run_dir / path.name))
                    self._note(f"profiler trace moved into the run folder: {path.name}")
                except OSError as e:
                    self._note(f"profiler trace {path.name} was left in {path.parent}: {e}")

    def _on_finished(self, exit_code: int, saw_oom: bool):
        if self._state == self.RENDERING:
            self._on_render_finished(exit_code)
            return
        self._close_log()
        self._proc = None
        self._collect_profiles()
        if self._canceled:
            self._set_state(self.CANCELED)
            self.canceled.emit()
            return
        if self._diverged is not None:
            self._finish_diverged(self._diverged)
            return
        if exit_code == 0 and Path(self._output_path).is_file():
            output_mtime = Path(self._output_path).stat().st_mtime
            if output_mtime >= self._spawn_stamp - 2:
                # A clean exit and a fresh file are not proof the run learned
                # anything, nor that the image tool can read what it wrote.
                # Bev_and_Billy (2026-08-17) satisfied both and was reported DONE.
                problem = verify.verify_output(self._output_path)
                if problem is None:
                    # The file is a LoRA. Is it the LoRA from the right moment?
                    # Hailey (2026-09-04) passed everything above and shipped the
                    # state from AFTER its loss had turned upward.
                    found = verify.trouble_from_log(str(self._run_dir / "trainer.log"))
                    if found is not None and not self._ship_snapshot(found):
                        self._discard_snapshots()
                        self._fail(
                            "Training ran to the end, but the LoRA it produced is not "
                            f"usable.\n\n{verify.describe(found)}\n\nNo snapshot "
                            f"from epoch {found['last_good_epoch']} or earlier was found "
                            "in the run's workspace\\save folder, so nothing could be "
                            f"shipped in its place. The file is still at "
                            f"{self._output_path} if you want to inspect it.")
                        return
                    self._discard_snapshots()
                    self._stamp_trained_words()
                    self._start_render()
                    return
                self._fail(
                    "Training ran to the end, but the LoRA it produced is not "
                    f"usable.\n\n{problem}\n\nThe file is still at "
                    f"{self._output_path} if you want to inspect it.")
                return
        if exit_code == 0:
            message = ("Training exited cleanly but produced no new output file — "
                       "check trainer.log in the run folder.")
        elif saw_oom:
            message = _OOM_HINT
        elif exit_code == ACCESS_VIOLATION:
            message = _CRASH_HINT + sysmem.biggest_consumer_note()
        else:
            message = f"Training failed (exit code {exit_code})."
        tail = "\n".join(self._tail)
        if tail:
            message += "\n\nLast trainer output:\n" + tail
        self._fail(message)
