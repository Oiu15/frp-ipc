from __future__ import annotations

"""Layer-boundary guard: services/length_service.py must not import tkinter."""

from pathlib import Path


_SERVICE_PATH = (
    Path(__file__).resolve().parents[1] / "services" / "length_service.py"
)


def _source() -> str:
    return _SERVICE_PATH.read_text(encoding="utf-8-sig")


class TestLengthServiceHasNoTkinter:
    def test_no_tkinter_import(self) -> None:
        source = _source()
        assert "import tkinter" not in source
        assert "from tkinter" not in source

    def test_no_messagebox_in_imports(self) -> None:
        source = _source()
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                assert "messagebox" not in stripped

    def test_no_stringvar_in_imports(self) -> None:
        source = _source()
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                assert "StringVar" not in stripped

    def test_no_apphost_import(self) -> None:
        source = _source()
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                assert "app_host" not in stripped.lower()
