from __future__ import annotations

import ast
from pathlib import Path
from uuid import uuid4

from tools.convert_unittest_to_pytest import convert_file


def _sample_path() -> Path:
    root = Path(".test-artifacts") / "converter-tool-tests"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"test_sample_{uuid4().hex}.py"


def test_converter_inserts_pytest_after_multiline_import_block() -> None:
    path = _sample_path()
    path.write_text(
        '''from pathlib import (
    Path,
)
import unittest


class SampleTest(unittest.TestCase):
    def test_value(self):
        self.assertAlmostEqual(Path("x").name.count("x"), 1)


if __name__ == '__main__':
    unittest.main()
''',
        encoding="utf-8",
    )

    assert convert_file(path)
    converted = path.read_text(encoding="utf-8")

    ast.parse(converted)
    assert "from pathlib import (\n    Path,\n)\nimport pytest\n" in converted
    assert "class TestSample:" in converted
    assert 'assert Path("x").name.count("x") == pytest.approx(1)' in converted
    assert "unittest.main" not in converted


def test_converter_preserves_edit_ranges_after_non_ascii_text() -> None:
    path = _sample_path()
    path.write_text(
        '''import unittest


class SampleTest(unittest.TestCase):
    def test_value(self):
        label = "中文前缀"
        self.assertEqual(f"{label}: ok", "中文前缀: ok")
''',
        encoding="utf-8",
    )

    assert convert_file(path)
    converted = path.read_text(encoding="utf-8")

    ast.parse(converted)
    assert 'assert f"{label}: ok" == "中文前缀: ok"' in converted


def test_converter_cleans_unittest_main_without_assertion_edits() -> None:
    path = _sample_path()
    path.write_text(
        '''import unittest


def test_already_pytest():
    assert True


if __name__ == '__main__':
    unittest.main()
''',
        encoding="utf-8",
    )

    assert convert_file(path)
    converted = path.read_text(encoding="utf-8")

    ast.parse(converted)
    assert "import unittest" not in converted
    assert "unittest.main" not in converted


def test_converter_rewrites_assert_raises_context_expr() -> None:
    path = _sample_path()
    path.write_text(
        '''import unittest


class SampleTest(unittest.TestCase):
    def test_value(self):
        with self.assertRaisesRegex(ValueError, "bad"):
            raise ValueError("bad value")
''',
        encoding="utf-8",
    )

    assert convert_file(path)
    converted = path.read_text(encoding="utf-8")

    ast.parse(converted)
    assert 'with pytest.raises(ValueError, match="bad"):' in converted
    assert "with with pytest.raises" not in converted
