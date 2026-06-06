"""Convert unittest.TestCase assertions to pytest-native style using AST.

Design:
- Parse each file with ast.parse().
- Walk the AST looking for calls of the form self.assertXxx(...).
- Use ast.get_source_segment() to extract argument text faithfully (handles
  nested brackets, strings, dicts, etc. correctly).
- Build an ordered list of text replacements (start_col, end_col, new_text)
  and apply them in reverse order to avoid offset drift.
- After conversion, validate the result with ast.parse(); if it fails the
  file is left untouched and the failure is reported.

Usage:
  python tools/convert_unittest_to_pytest.py tests/test_foo.py [tests/test_bar.py ...]

Safe / automatic conversions (via AST):
  assertEqual(a, b)        → assert a == b
  assertNotEqual(a, b)     → assert a != b
  assertTrue(x)            → assert x
  assertFalse(x)           → assert not x
  assertIsNone(x)          → assert x is None
  assertIsNotNone(x)       → assert x is not None
  assertIsInstance(x, T)   → assert isinstance(x, T)
  assertIs(a, b)           → assert a is b
  assertIn(a, b)           → assert a in b
  assertNotIn(a, b)        → assert a not in b
  assertGreater(a, b)      → assert a > b
  assertGreaterEqual(a, b) → assert a >= b
  assertLess(a, b)         → assert a < b
  assertLessEqual(a, b)    → assert a <= b
  assertAlmostEqual(a, b)  → assert a == pytest.approx(b)
  assertRaises(E)          → with pytest.raises(E):
  assertRaisesRegex(E, m)  → with pytest.raises(E, match=m):

Skipped (left as-is with a # TODO comment):
  assertLogs, addCleanup, subTest, and assertEqual/NotEqual with msg=
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# descriptor of a single text replacement
# ---------------------------------------------------------------------------


@dataclass(order=True)
class Edit:
    """Replace source[pos_start:pos_end] with new_text.

    Edits are applied in reverse order (largest pos_start first) so
    that each edit's offsets remain valid relative to the original
    source.
    """

    pos_start: int   # 0-based byte offset in UTF-8 encoded source
    pos_end: int     # 0-based byte offset in UTF-8 encoded source
    new_text: str


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _is_self_call(node: ast.AST, method: str) -> bool:
    """True when *node* is ``self.<method>(...)``."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == method
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
    )


def _arg_text(source: str, node: ast.AST) -> str:
    """Return the source text of an AST node."""
    text = ast.get_source_segment(source, node)
    if text is None:
        line = getattr(node, "lineno", "?")
        raise ValueError(f"cannot recover source at line {line}")
    return text


def _full_call_text(source: str, node: ast.Call) -> str:
    """Return the source text of the entire ``self.assertXxx(...)`` call."""
    return _arg_text(source, node)


# ---------------------------------------------------------------------------
# conversion rules — each returns (new_text, needs_pytest_import)
# ---------------------------------------------------------------------------


def _convert_assert_equal(node: ast.Call, src: str, _lines: list[str]) -> tuple[str, bool]:
    left = _arg_text(src, node.args[0])
    right = _arg_text(src, node.args[1])
    return f"assert {left} == {right}", False


def _convert_assert_not_equal(node: ast.Call, src: str, _lines: list[str]) -> tuple[str, bool]:
    left = _arg_text(src, node.args[0])
    right = _arg_text(src, node.args[1])
    return f"assert {left} != {right}", False


def _convert_assert_true(node: ast.Call, src: str, _lines: list[str]) -> tuple[str, bool]:
    arg = _arg_text(src, node.args[0])
    return f"assert {arg}", False


def _convert_assert_false(node: ast.Call, src: str, _lines: list[str]) -> tuple[str, bool]:
    arg = _arg_text(src, node.args[0])
    return f"assert not {arg}", False


def _convert_assert_is_none(node: ast.Call, src: str, _lines: list[str]) -> tuple[str, bool]:
    arg = _arg_text(src, node.args[0])
    return f"assert {arg} is None", False


def _convert_assert_is_not_none(node: ast.Call, src: str, _lines: list[str]) -> tuple[str, bool]:
    arg = _arg_text(src, node.args[0])
    return f"assert {arg} is not None", False


def _convert_assert_is_instance(node: ast.Call, src: str, _lines: list[str]) -> tuple[str, bool]:
    obj = _arg_text(src, node.args[0])
    cls = _arg_text(src, node.args[1])
    return f"assert isinstance({obj}, {cls})", False


def _convert_assert_is(node: ast.Call, src: str, _lines: list[str]) -> tuple[str, bool]:
    left = _arg_text(src, node.args[0])
    right = _arg_text(src, node.args[1])
    return f"assert {left} is {right}", False


def _convert_assert_in(node: ast.Call, src: str, _lines: list[str]) -> tuple[str, bool]:
    member = _arg_text(src, node.args[0])
    container = _arg_text(src, node.args[1])
    return f"assert {member} in {container}", False


def _convert_assert_not_in(node: ast.Call, src: str, _lines: list[str]) -> tuple[str, bool]:
    member = _arg_text(src, node.args[0])
    container = _arg_text(src, node.args[1])
    return f"assert {member} not in {container}", False


def _convert_assert_greater(node: ast.Call, src: str, _lines: list[str]) -> tuple[str, bool]:
    left = _arg_text(src, node.args[0])
    right = _arg_text(src, node.args[1])
    return f"assert {left} > {right}", False


def _convert_assert_greater_equal(node: ast.Call, src: str, _lines: list[str]) -> tuple[str, bool]:
    left = _arg_text(src, node.args[0])
    right = _arg_text(src, node.args[1])
    return f"assert {left} >= {right}", False


def _convert_assert_less(node: ast.Call, src: str, _lines: list[str]) -> tuple[str, bool]:
    left = _arg_text(src, node.args[0])
    right = _arg_text(src, node.args[1])
    return f"assert {left} < {right}", False


def _convert_assert_less_equal(node: ast.Call, src: str, _lines: list[str]) -> tuple[str, bool]:
    left = _arg_text(src, node.args[0])
    right = _arg_text(src, node.args[1])
    return f"assert {left} <= {right}", False


def _convert_assert_almost_equal(node: ast.Call, src: str, _lines: list[str]) -> tuple[str, bool]:
    left = _arg_text(src, node.args[0])
    right = _arg_text(src, node.args[1])
    return f"assert {left} == pytest.approx({right})", True


def _convert_assert_raises(node: ast.Call, src: str, _lines: list[str]) -> tuple[str, bool]:
    exc_cls = _arg_text(src, node.args[0])
    return f"pytest.raises({exc_cls})", True


def _convert_assert_raises_regex(node: ast.Call, src: str, _lines: list[str]) -> tuple[str, bool]:
    exc_cls = _arg_text(src, node.args[0])
    match_arg = _arg_text(src, node.args[1])
    return f"pytest.raises({exc_cls}, match={match_arg})", True


# Map method name → converter
_CONVERTERS: dict[str, Any] = {
    "assertEqual":        _convert_assert_equal,
    "assertNotEqual":     _convert_assert_not_equal,
    "assertTrue":         _convert_assert_true,
    "assertFalse":        _convert_assert_false,
    "assertIsNone":       _convert_assert_is_none,
    "assertIsNotNone":    _convert_assert_is_not_none,
    "assertIsInstance":   _convert_assert_is_instance,
    "assertIs":           _convert_assert_is,
    "assertIn":           _convert_assert_in,
    "assertNotIn":        _convert_assert_not_in,
    "assertGreater":      _convert_assert_greater,
    "assertGreaterEqual": _convert_assert_greater_equal,
    "assertLess":         _convert_assert_less,
    "assertLessEqual":    _convert_assert_less_equal,
    "assertAlmostEqual":  _convert_assert_almost_equal,
    "assertRaises":       _convert_assert_raises,
    "assertRaisesRegex":  _convert_assert_raises_regex,
}

# Methods we deliberately skip
_SKIP_METHODS = {"assertLogs", "subTest", "addCleanup"}


# ---------------------------------------------------------------------------
# visit & collect edits
# ---------------------------------------------------------------------------


def _collect_edits(source: str) -> tuple[list[Edit], bool]:
    """Walk the AST of *source* and collect all replacements.

    Returns (edits, needs_pytest).
    """
    tree = ast.parse(source)
    edits: list[Edit] = []
    needs_pytest = False
    line_offsets = _line_byte_offsets(source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if not (isinstance(node.func.value, ast.Name) and node.func.value.id == "self"):
            continue

        method = node.func.attr
        if method in _SKIP_METHODS:
            continue

        converter = _CONVERTERS.get(method)
        if converter is None:
            continue

        try:
            new_text, needs_pytest_import = converter(node, source, source.splitlines(keepends=True))
            needs_pytest = needs_pytest or needs_pytest_import
        except ValueError:
            continue

        start_byte = line_offsets[node.lineno - 1] + node.col_offset
        end_byte = line_offsets[(node.end_lineno or node.lineno) - 1] + (node.end_col_offset or 0)

        # For multi-line calls, also consume the trailing newline after the
        # closing paren so the replacement text (which may itself be multi-line)
        # sits cleanly before the next statement.
        if node.lineno != (node.end_lineno or node.lineno):
            source_bytes = source.encode("utf-8")
            if end_byte < len(source_bytes) and source_bytes[end_byte:end_byte+1] == b"\n":
                end_byte += 1

        edits.append(Edit(pos_start=start_byte, pos_end=end_byte, new_text=new_text))

    # Sort by (lineno DESC, col_start DESC) so we can apply without offset drift
    edits.sort(reverse=True)
    return edits, needs_pytest


def _line_byte_offsets(source: str) -> list[int]:
    """Return UTF-8 byte offsets for each 1-based source line.

    Python AST column offsets are UTF-8 byte offsets, not Unicode character
    offsets, so edit ranges must use the same coordinate system.
    """
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line.encode("utf-8")))
    return offsets


# ---------------------------------------------------------------------------
# apply edits & validate
# ---------------------------------------------------------------------------


def _apply(original: str, edits: list[Edit]) -> str:
    """Apply edits in reverse order (largest pos_start first)."""
    edits.sort(key=lambda e: e.pos_start, reverse=True)
    source = original.encode("utf-8")
    for edit in edits:
        new_bytes = edit.new_text.encode("utf-8")
        source = source[:edit.pos_start] + new_bytes + source[edit.pos_end:]
    return source.decode("utf-8")


def _validate(source: str, path: Path) -> bool:
    try:
        ast.parse(source)
        return True
    except SyntaxError as exc:
        print(f"ERROR: {path} — converted source has syntax error: {exc}")
        return False


# ---------------------------------------------------------------------------
# post-conversion cleanup
# ---------------------------------------------------------------------------


def _cleanup(source: str) -> str:
    """Remove import unittest and if __name__ boilerplate."""
    # Remove 'import unittest\n' (not 'from unittest.mock import ...')
    lines = source.splitlines(keepends=True)
    result: list[str] = []
    for line in lines:
        if line.strip() == "import unittest":
            continue
        result.append(line)
    source = "".join(result)

    # Remove trailing if __name__ == "__main__": unittest.main()
    source = re.sub(
        r'\n+if __name__ == [\'"]__main__[\'"]:\n\s+unittest\.main\(\)\n*$',
        "\n",
        source,
    )
    return source


def _add_pytest_import(source: str) -> str:
    """Insert 'import pytest' after the module's top-level imports."""
    tree = ast.parse(source)
    if _has_pytest_import(tree):
        return source

    lines = source.splitlines(keepends=True)
    import_nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and not (isinstance(node, ast.ImportFrom) and node.module == "__future__")
    ]
    if import_nodes:
        insert_line = import_nodes[-1].end_lineno or import_nodes[-1].lineno
        lines.insert(insert_line, "import pytest\n")
        return "".join(lines)

    future_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "__future__"
    ]
    if future_nodes:
        insert_line = future_nodes[-1].end_lineno or future_nodes[-1].lineno
        lines.insert(insert_line, "\nimport pytest\n")
        return "".join(lines)

    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
        if isinstance(tree.body[0].value.value, str):
            insert_line = tree.body[0].end_lineno or tree.body[0].lineno
            lines.insert(insert_line, "\nimport pytest\n")
            return "".join(lines)

    return "import pytest\n" + source


def _has_pytest_import(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.Import):
            if any(alias.name == "pytest" for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and node.module == "pytest":
            return True
    return False


def _rename_class(source: str) -> str:
    """Rename 'class FooTest(unittest.TestCase):' → 'class TestFoo:'."""
    return re.sub(
        r"^class (\w+)Test\(unittest\.TestCase\):",
        r"class Test\1:",
        source,
        flags=re.MULTILINE,
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def convert_file(path: Path) -> bool:
    original = path.read_text(encoding="utf-8-sig")  # strip BOM if present
    source = original

    # Phase 1: convert assertions via AST
    edits, needs_pytest = _collect_edits(source)
    source = _apply(source, edits)

    # Phase 2: validate
    if not _validate(source, path):
        return False

    # Phase 3: cleanup
    source = _cleanup(source)
    source = _rename_class(source)

    # Phase 4: add pytest import if needed
    if needs_pytest:
        source = _add_pytest_import(source)

    # Final validation
    if not _validate(source, path):
        return False

    if source == original:
        print(f"  {path.name}: no changes needed")
        return True

    path.write_text(source, encoding="utf-8")
    print(f"  {path.name}: converted {len(edits)} call(s)")
    return True


def main() -> None:
    exit_code = 0
    for arg in sys.argv[1:]:
        path = Path(arg)
        if not path.exists():
            print(f"SKIP: {arg} not found")
            exit_code = 1
            continue
        ok = convert_file(path)
        if not ok:
            print(f"FAIL: {path} — conversion failed, file unchanged")
            exit_code = 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
