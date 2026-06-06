from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys
import unittest


class RepositoryImportBoundariesTest(unittest.TestCase):
    def test_recipe_repository_import_does_not_load_other_repositories(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import repositories.recipe_repository; "
                    "assert 'repositories.calibration_repository' not in sys.modules; "
                    "assert 'repositories.run_repository' not in sys.modules; "
                    "assert 'repositories.validation_repository' not in sys.modules; "
                    "assert 'numpy' not in sys.modules"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_repositories_do_not_import_services(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        offenders: list[str] = []
        for path in (repo_root / "repositories").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "services":
                    offenders.append(f"{path.name}:{node.lineno}: from {node.module} import ...")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] == "services":
                            offenders.append(f"{path.name}:{node.lineno}: import {alias.name}")

        self.assertEqual(offenders, [])

    def test_run_repository_export_without_index_writer_does_not_load_services(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import shutil, sys; "
                    "from pathlib import Path; "
                    "from domain.state import CalibrationSnapshot, RunContext, RunIdentity; "
                    "from core.models import Recipe; "
                    "from repositories.run_repository import RunRepository; "
                    "root = Path('.test-artifacts') / 'repository_import_boundary' / 'FRP_IPC'; "
                    "shutil.rmtree(root.parent, ignore_errors=True); "
                    "ctx = RunContext("
                    "identity=RunIdentity(serial='s', run_id='r', started_at_ts=1735787045.0), "
                    "recipe=Recipe(name='r'), calibration=CalibrationSnapshot(), rows=[], raw_points=[], summary={}, "
                    "finished_at_ts=1735787046.0, status='STOP', completed=False"
                    "); "
                    "RunRepository(app_root_dir=root).export_run(ctx); "
                    "assert not any(m.startswith('services') for m in sys.modules), "
                    "[m for m in sys.modules if m.startswith('services')]"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
