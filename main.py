import sys
import os
import warnings

# Suppress the xformers/torch impl_abstract → register_fake rename warning
warnings.filterwarnings("ignore", message=".*impl_abstract.*", category=FutureWarning)

# Ensure the project root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from PySide6.QtCore import Qt

from utils.styles import DARK_STYLESHEET
from ui.main_window import MainWindow


def main():
    # Enable high-DPI scaling on Windows
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

    app = QApplication(sys.argv)
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

    # Apply dark stylesheet globally
    app.setStyleSheet(DARK_STYLESHEET)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
