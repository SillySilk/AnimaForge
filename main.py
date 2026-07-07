import argparse
import sys
import os
import warnings

# Suppress the xformers/torch impl_abstract → register_fake rename warning
warnings.filterwarnings("ignore", message=".*impl_abstract.*", category=FutureWarning)

# Ensure the project root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def build_app(argv=None):
    """Construct the QApplication and MainWindow exactly as the app boots — fonts,
    stylesheet, font preference, settings migration, icon — but do NOT show() or exec().

    Returns (app, window). Shared by main() and tooling (e.g. scripts/capture_samples.py)
    so screenshots and headless drivers use the identical look the real app renders.
    """
    # GUI-only imports live here so --run-batch (core.headless) never pays
    # for, or depends on, the ui stack.
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QIcon

    from utils.fonts import load_app_fonts
    from ui.main_window import MainWindow

    # Enable high-DPI scaling on Windows
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

    app = QApplication(sys.argv if argv is None else argv)
    app.setApplicationName("AnimaForge")
    app.setApplicationDisplayName("Anima Forge LoRA Trainer")
    app.setOrganizationName("AnimaForge")
    app.setOrganizationDomain("animaforge.local")

    # One-time, lossless migration of the old PonyExpress/LoRATrainer QSettings store.
    from core.settings import migrate_legacy_settings
    migrate_legacy_settings()

    # App / taskbar icon (the forge emblem)
    _icon = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icon.ico")
    if os.path.exists(_icon):
        app.setWindowIcon(QIcon(_icon))

    # Register the bundled forge fonts, then apply the stylesheet with the user's
    # font preference resolved (forge faces by default; see utils/fonts.py).
    from utils.fonts import apply_app_font
    load_app_fonts()
    apply_app_font(app)

    window = MainWindow()
    return app, window


def main():
    if "--run-batch" in sys.argv[1:]:
        parser = argparse.ArgumentParser(prog="AnimaForge headless")
        parser.add_argument("--run-batch", action="store_true")
        parser.add_argument("--queue", default=None,
                            help="queue file (default: batch_queue.json)")
        parser.add_argument("--status", default=None,
                            help="status file (default: <output dir>/batch_status.json)")
        args = parser.parse_args()
        from core.headless import run_headless
        sys.exit(run_headless(args.queue, args.status))

    app, window = build_app()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
