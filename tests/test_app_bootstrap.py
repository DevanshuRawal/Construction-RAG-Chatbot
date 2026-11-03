import importlib.machinery
from pathlib import Path
import sys

from src.path_setup import ensure_project_root
from src.chat_ui import startup_failure_message


def test_project_root_is_added_when_streamlit_starts_in_src_directory(monkeypatch):
    project_root = Path(__file__).resolve().parents[1]
    src_directory = project_root / "src"
    existing_paths = [
        path
        for path in sys.path
        if not path or Path(path).resolve() != project_root
    ]
    monkeypatch.setattr(sys, "path", [str(src_directory), *existing_paths])

    ensure_project_root()

    package_spec = importlib.machinery.PathFinder.find_spec("src", sys.path)
    assert package_spec is not None
    assert Path(package_spec.origin).resolve() == project_root / "src" / "__init__.py"


def test_startup_error_message_does_not_expose_exception_details():
    message = startup_failure_message()

    assert "could not start" in message.lower()
    assert "traceback" not in message.lower()
    assert "api_key" not in message.lower()
