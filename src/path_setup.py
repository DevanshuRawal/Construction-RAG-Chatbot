"""Import path helpers for running the app from its entrypoint directory."""
from pathlib import Path
import sys


def ensure_project_root():
    """Make the repository root importable when app.py is run from ``src``."""
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
