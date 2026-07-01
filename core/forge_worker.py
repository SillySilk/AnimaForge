"""QThread worker that test-renders a LoRA through Forge's txt2img API.

Lives off the UI thread because each render takes seconds. Delivery (a file copy) is fast and
done inline by callers via forge_api.deliver_lora — no worker needed for that.
"""
from PySide6.QtCore import QObject, Signal

from core import forge_api


class ForgeRenderWorker(QObject):
    image_ready = Signal(bytes, int)   # png bytes, prompt index
    log_line = Signal(str)
    finished = Signal(bool)            # True if at least one image came back

    def __init__(self, api_url: str, lora_name: str, trigger: str, prompts, steps: int = 24, parent=None):
        super().__init__(parent)
        self._url = api_url
        self._lora = lora_name
        self._trigger = trigger
        self._prompts = prompts
        self._steps = steps

    def run(self):
        if not forge_api.ping(self._url):
            self.log_line.emit(f"[Forge] Cannot reach {self._url} — start Forge with --api enabled.")
            self.finished.emit(False)
            return
        got_any = False
        for i, prompt in enumerate(self._prompts):
            try:
                payload = forge_api.build_test_payload(self._lora, self._trigger, prompt, steps=self._steps)
                self.log_line.emit(f"[Forge] Rendering {i + 1}/{len(self._prompts)}: {prompt[:50]}")
                for img in forge_api.txt2img(self._url, payload):
                    self.image_ready.emit(img, i)
                    got_any = True
            except Exception as e:
                self.log_line.emit(f"[Forge] render error on prompt {i + 1}: {e}")
        self.finished.emit(got_any)
