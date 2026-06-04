from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
