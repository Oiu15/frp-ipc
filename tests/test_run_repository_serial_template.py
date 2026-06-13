import datetime
import json
import shutil
import time
from pathlib import Path

from repositories.run_repository import RunRepository
from repositories.settings_repository import SettingsRepository


def _today() -> str:
    return datetime.date.today().strftime("%Y%m%d")


def _case_root(name: str) -> Path:
    root = Path(__file__).resolve().parents[1] / ".compile_check" / "run_repository_serial_template"
    path = root / f"{name}_{int(time.time() * 1000)}"
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_prepare_run_default_template_uses_legacy_counter_shape() -> None:
    app_root = _case_root("default")
    repo = RunRepository(app_root_dir=app_root)

    identity = repo.prepare_run("default_recipe")

    assert identity.serial == f"{_today()}-default_recipe-001"
    counters = json.loads((app_root / "run_counter.json").read_text(encoding="utf-8"))
    assert counters[_today()]["default_recipe"] == 1
    shutil.rmtree(app_root, ignore_errors=True)


def test_prepare_run_custom_template_updates_only_target_counter_key() -> None:
    app_root = _case_root("custom")
    settings_repo = SettingsRepository(app_root_dir=app_root)
    settings_repo.save_serial_template(
        {
            "separator": "-",
            "custom_values": {"customer": "客户A"},
            "fields": [
                {"type": "system", "key": "date", "enabled": True},
                {"type": "custom", "key": "customer", "enabled": True},
                {"type": "system", "key": "seq", "enabled": True},
            ],
        }
    )
    day = _today()
    (app_root / "run_counter.json").write_text(
        json.dumps({day: {"legacy_recipe": 7}, "20250101": {"old": 2}}, ensure_ascii=False),
        encoding="utf-8",
    )
    repo = RunRepository(app_root_dir=app_root)

    first = repo.prepare_run("ignored")
    second = repo.prepare_run("ignored")

    assert first.serial == f"{day}-客户A-001"
    assert second.serial == f"{day}-客户A-002"
    counters = json.loads((app_root / "run_counter.json").read_text(encoding="utf-8"))
    assert counters[day]["客户A"] == 2
    assert counters[day]["legacy_recipe"] == 7
    assert counters["20250101"]["old"] == 2
    shutil.rmtree(app_root, ignore_errors=True)


def test_prepare_run_without_seq_uses_suffix_and_does_not_create_counter() -> None:
    app_root = _case_root("without_seq")
    SettingsRepository(app_root_dir=app_root).save_serial_template(
        {
            "separator": "-",
            "custom_values": {"customer": "客户A"},
            "fields": [
                {"type": "system", "key": "date", "enabled": True},
                {"type": "custom", "key": "customer", "enabled": True},
            ],
        }
    )
    repo = RunRepository(app_root_dir=app_root)

    identity = repo.prepare_run("ignored")

    assert identity.serial.startswith(f"{_today()}-客户A__")
    assert len(identity.serial.rsplit("__", 1)[1]) == 6
    assert not (app_root / "run_counter.json").exists()
    shutil.rmtree(app_root, ignore_errors=True)
