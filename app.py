"""
app.py — Application Launcher

Launches the Audio Signal Analyzer and Noise Reduction System Using FFT.
"""

import sys
from PySide6.QtWidgets import QApplication
from gui import MainWindow


def run_app():
    """Initialize and run the desktop application."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    app.setApplicationName("Audio Signal Analyzer & Noise Reduction")
    app.setOrganizationName("SNS Academic Project")

    # Set default application font to prevent negative point size warnings
    from PySide6.QtGui import QFont
    app.setFont(QFont("Segoe UI", 10))

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(run_app())
