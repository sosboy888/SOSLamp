"""App entrypoint."""

from __future__ import annotations

import sys
import logging

from PySide6 import QtWidgets

from soslamp.gui.main_window import SOSLampWindow
from soslamp.utils.async_bridge import AsyncExecutor


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    app = QtWidgets.QApplication(sys.argv)
    executor = AsyncExecutor()
    window = SOSLampWindow(executor)
    window.show()
    code = app.exec()
    executor.stop()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
