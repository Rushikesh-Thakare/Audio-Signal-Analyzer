"""
tests/verify_launch.py — Verifies that app.py initializes and can start without crashing.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from gui import MainWindow


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()

    print("[SUCCESS] MainWindow created and displayed without runtime exceptions.")

    # Automatically close after 1000 ms to confirm non-blocking shutdown
    QTimer.singleShot(1000, app.quit)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
