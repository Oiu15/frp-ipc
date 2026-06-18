from __future__ import annotations

"""Layer-boundary guard: services/teach_service.py must not import tkinter."""

from pathlib import Path


_SERVICE_PATH = (
    Path(__file__).resolve().parents[1] / "services" / "teach_service.py"
)


def _source() -> str:
    return _SERVICE_PATH.read_text(encoding="utf-8-sig")


class TestTeachServiceHasNoTkinter:
    def test_no_tkinter_import(self) -> None:
        source = _source()
        assert "import tkinter" not in source
        assert "from tkinter" not in source

    def test_no_messagebox_import(self) -> None:
        """messagebox must not appear in import lines."""
        source = _source()
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                assert "messagebox" not in stripped, (
                    f"messagebox found in import: {stripped}"
                )

    def test_no_stringvar_in_imports(self) -> None:
        source = _source()
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                assert "StringVar" not in stripped, (
                    f"StringVar found in import: {stripped}"
                )

    def test_no_treeview_in_imports(self) -> None:
        source = _source()
        for line in source.splitlines():
            stripped = line.strip()
            if (stripped.startswith("import ") or stripped.startswith("from ")):
                assert "Treeview" not in stripped, (
                    f"Treeview found: {stripped}"
                )

    def test_no_apphost_import(self) -> None:
        source = _source()
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                assert "app_host" not in stripped.lower(), (
                    f"app_host found in import: {stripped}"
                )
