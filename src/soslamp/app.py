"""App entrypoint."""

from __future__ import annotations

import sys
import logging

from PySide6 import QtWidgets

from soslamp.gui.main_window import SOSLampWindow
from soslamp.utils.async_bridge import AsyncExecutor


def main() -> int:
    # Set up logging first so every module (BLE, audio, GUI) can use logging.getLogger(__name__).
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Qt needs exactly one QApplication before any widget is created.
    app = QtWidgets.QApplication(sys.argv)
    # Bluetooth (bleak) is async but Qt is not, so the async code runs on its own thread.
    executor = AsyncExecutor()
    window = SOSLampWindow(executor)
    window.show()
    # app.exec() blocks until the window closes and gives back an exit code.
    code = app.exec()
    # Shut the background event loop down so the process can exit cleanly.
    executor.stop()
    return code


# Only run when started directly (python -m soslamp.app), not when imported.
if __name__ == "__main__":
    raise SystemExit(main())
