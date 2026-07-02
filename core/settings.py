"""Central app configuration over QSettings.

One typed home for every configurable value (paths, connections, app defaults, and the
Anima-honored advanced training knobs), replacing scattered ad-hoc QSettings calls and the
hardcoded Forge-Neo path. `build_extra_training_args()` / `prepare_sample_args()` turn the
advanced settings into the dict that `config_generator.generate_configs(extra_args=...)` merges.
"""
from pathlib import Path

from PySide6.QtCore import QSettings

SETTINGS_ORG = "AnimaForge"
SETTINGS_APP = "AnimaForge"

# Legacy store (the project's old working name) — migrated once into the new store.
LEGACY_ORG = "PonyExpress"
LEGACY_APP = "LoRATrainer"
_MIGRATED_KEY = "_migrated_from_pony"


def _migrate_between(old: QSettings, new: QSettings) -> None:
    """Pure: copy keys from the legacy store into the new one, once, losslessly.

    No-op if already migrated or if the new store already has data (never overwrites).
    """
    if new.value(_MIGRATED_KEY, False, type=bool):
        return
    if old.allKeys() and not new.allKeys():
        for k in old.allKeys():
            new.setValue(k, old.value(k))
    new.setValue(_MIGRATED_KEY, True)
    new.sync()


def migrate_legacy_settings() -> None:
    """One-time migration of the PonyExpress/LoRATrainer store into AnimaForge."""
    _migrate_between(QSettings(LEGACY_ORG, LEGACY_APP), QSettings(SETTINGS_ORG, SETTINGS_APP))

DEFAULTS = {
    # Paths
    "model_scan_dir": "",
    "sdscripts_path": "",
    "dit_path": "",
    "qwen3_path": "",
    "vae_path": "",
    "output_dir": "",
    # Connections
    "lmstudio_url": "http://localhost:1234/v1",
    # Recommended public vision model; leave blank to use whatever LM Studio has loaded.
    "lmstudio_model": "qwen2.5-vl-7b-instruct",
    "forge_api_url": "http://127.0.0.1:7860",
    "forge_lora_dir": "",
    "forge_auto_deliver": False,
    "forge_auto_test": False,
    # ComfyUI has no test-render API we can target (workflows vary), but a plain
    # copy into its LoRA folder covers the ComfyUI crowd (user feedback).
    "comfyui_lora_dir": "",
    # Custom training presets (JSON array — see core/train_presets.py)
    "train_presets_json": "",
    # App defaults (Train tab initializes from these)
    "default_optimizer": "prodigy",
    "default_network_dim": 16,
    "default_network_alpha": 8,
    "default_target_steps": 500,
    # Power-user escape hatch: when True the auto step suggestion ignores the soft cap
    # (SOFT_CAP_STEPS) so large datasets can train longer. The floor still applies.
    "default_uncap_steps": False,
    "default_caption_order": "nl_first",
    "default_train_text_encoder": False,
    # Advanced training (Anima-honored only)
    "weighting_scheme": "sigmoid",
    "logit_mean": 0.0,
    "logit_std": 1.0,
    "caption_dropout_rate": 0.0,
    "flip_aug": False,
    "network_dropout": 0.0,
    # Resumability: write a training state every N steps (0 = epoch boundaries only)
    # so a mid-epoch Stop can resume losing at most this many steps.
    "save_every_n_steps": 250,
    # Pre-launch guard: warn if free VRAM is below this many MB before training (0 disables).
    "min_free_vram_mb": 12000,
    # Sample images during training (on by default so progress previews "just happen").
    # Every epoch by default: the workflow is watching per-epoch previews and stopping
    # at the earliest-best — a sparse default hid progress for whole runs (user feedback).
    "sample_enable": True,
    "sample_prompts": "",
    "sample_every_n_epochs": 1,
    "sample_count": 4,
    "sample_sampler": "euler_a",
    "sample_at_first": True,
    # Quality/safety scaffolding prepended to every preview prompt (after the trigger/anchor)
    # so progress images don't render in Anima's plain "unrefined base" style. Editable in
    # Setup; clear it to preview raw LoRA output. Lowercase, score_* keeps its underscore.
    "sample_quality_prefix": "masterpiece, best quality, score_7, safe",
}

# Note: fp8_scaled is intentionally absent — anima_train_network.py hard-disables it
# ("Anima DiT does not support fp8_scaled"), so exposing it would be a dead control.


# Keys held in memory for the session only — never written to the backing store, so
# they start fresh each app launch instead of carrying last session's value over. The
# sample-prompts box is auto-filled from the dataset, so persisting it just left stale text.
EPHEMERAL_KEYS = {"sample_prompts"}


class AppSettings:
    def __init__(self, settings: QSettings = None):
        self._s = settings or QSettings(SETTINGS_ORG, SETTINGS_APP)
        # Session-only store for EPHEMERAL_KEYS; shared across tabs because they share
        # this AppSettings instance, but gone on restart since it never hits disk.
        self._ephemeral = {}

    def get(self, key: str):
        default = DEFAULTS[key]
        if key in EPHEMERAL_KEYS:
            val = self._ephemeral.get(key, default)
        else:
            val = self._s.value(key, default)
        if isinstance(default, bool):
            if isinstance(val, str):
                return val.strip().lower() in ("true", "1", "yes")
            return bool(val)
        if isinstance(default, int):
            try:
                return int(val)
            except (TypeError, ValueError):
                return default
        if isinstance(default, float):
            try:
                return float(val)
            except (TypeError, ValueError):
                return default
        return str(val) if val is not None else default

    def set(self, key: str, value) -> None:
        if key in EPHEMERAL_KEYS:
            self._ephemeral[key] = value
            return
        self._s.setValue(key, value)
        self._s.sync()

    def first_run_scan_default(self) -> str:
        """Prefill for the model scan folder: saved value, else a generic auto-guess."""
        saved = self.get("model_scan_dir")
        if saved:
            return saved
        from core.model_locations import guess_model_scan_dir
        return guess_model_scan_dir()

    def build_extra_training_args(self) -> dict:
        """Advanced training-block args, omitting defaults/disabled (no dead keys)."""
        e = {}
        if self.get("weighting_scheme") != "sigmoid":
            e["weighting_scheme"] = self.get("weighting_scheme")
        if self.get("logit_mean") != 0.0:
            e["logit_mean"] = self.get("logit_mean")
        if self.get("logit_std") != 1.0:
            e["logit_std"] = self.get("logit_std")
        if self.get("caption_dropout_rate") > 0:
            e["caption_dropout_rate"] = self.get("caption_dropout_rate")
        if self.get("flip_aug"):
            e["flip_aug"] = True
        if self.get("network_dropout") > 0:
            e["network_dropout"] = self.get("network_dropout")
        return e

    def prepare_sample_args(self, output_dir: str, lora_name: str, trigger_word: str = "",
                            style_anchor: str = "") -> dict:
        """If sample images are enabled, write the prompts file and return sample-* training args.

        Activation words (the trigger and the style anchor, if any) are prepended to every prompt
        so previews actually exercise the trained concept/style — each only added if not already
        present as a comma-separated token.
        """
        if not self.get("sample_enable"):
            return {}
        activations = [a.strip() for a in (trigger_word, style_anchor) if a and a.strip()]
        quality = [q.strip() for q in self.get("sample_quality_prefix").split(",") if q.strip()]
        prompts = [p.strip() for p in self.get("sample_prompts").splitlines() if p.strip()]
        if not prompts:
            # Enabled but no prompts authored yet: fall back to a minimal preview so
            # progress images still render (activation words alone exercise likeness/style).
            prompts = [", ".join(activations)] if activations else ["portrait, high quality"]
        lines = []
        for p in prompts:
            existing = {t.strip().lower() for t in p.split(",") if t.strip()}
            # Canonical Anima order: quality/safety scaffolding first, then the learned
            # activations (trigger + @style anchor), then the prompt body. Skip duplicates.
            prefix = []
            for tok in quality + activations:
                if tok.lower() not in existing:
                    prefix.append(tok)
                    existing.add(tok.lower())
            if prefix:
                p = ", ".join(prefix + [p])
            lines.append(p)
        cfg_dir = Path(output_dir) / "configs"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        f = cfg_dir / f"{lora_name}_sample.txt"
        f.write_text("\n".join(lines) + "\n", encoding="utf-8")
        args = {
            "sample_prompts": str(f.as_posix()),
            "sample_every_n_epochs": self.get("sample_every_n_epochs"),
            "sample_sampler": self.get("sample_sampler"),
        }
        if self.get("sample_at_first"):
            args["sample_at_first"] = True
        return args
