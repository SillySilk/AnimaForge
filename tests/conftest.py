"""Shared pytest setup.

Tear down every widget a finished test module left behind. Test functions build
top-level widgets (whole MainWindows, tabs, dialogs) and drop them, but Qt keeps them
alive from the C++ side, so they pile up across the suite. Any test that sets an
application-wide stylesheet (utils.fonts.apply_app_font, via the font tests and the
Setup tab's font mode) re-polishes EVERY live widget in that pile, and doing so
intermittently died with a Windows access violation. Krea Forge diagnosed this
(~1 run in 8) and fixed it in its own runner; this brings the same teardown to pytest.
"""
import pytest


def _destroy_leaked_widgets():
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        return
    for w in QApplication.topLevelWidgets():
        w.close()
        w.deleteLater()
    # deleteLater only queues the destruction; drain it here, not later inside
    # another test's widget walk.
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    QCoreApplication.processEvents()


@pytest.fixture(autouse=True, scope="module")
def _destroy_widgets_after_each_module():
    yield
    _destroy_leaked_widgets()
