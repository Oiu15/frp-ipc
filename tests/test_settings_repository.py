import json
import shutil
import time
from pathlib import Path

from repositories.settings_repository import SettingsRepository


def _case_root(name: str) -> Path:
    root = Path(__file__).resolve().parents[1] / ".compile_check" / "settings_repository"
    path = root / f"{name}_{int(time.time() * 1000)}"
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_missing_settings_returns_default_template() -> None:
    app_root = _case_root("missing")
    repo = SettingsRepository(app_root_dir=app_root)

    settings = repo.load_settings()

    assert settings["serial_template"]["fields"][-1]["key"] == "seq"
    shutil.rmtree(app_root, ignore_errors=True)


def test_save_settings_uses_final_file_without_temp_leftover() -> None:
    app_root = _case_root("save")
    repo = SettingsRepository(app_root_dir=app_root)
    settings = repo.default_settings()
    settings["serial_template"]["custom_values"]["customer"] = "客户A"

    repo.save_settings(settings)

    payload = json.loads((app_root / "settings.json").read_text(encoding="utf-8"))
    assert payload["serial_template"]["custom_values"]["customer"] == "客户A"
    assert not (app_root / "settings.json.tmp").exists()
    shutil.rmtree(app_root, ignore_errors=True)


def test_corrupt_settings_is_backed_up_and_defaulted() -> None:
    app_root = _case_root("corrupt")
    settings_path = app_root / "settings.json"
    settings_path.write_text("{bad json", encoding="utf-8")
    repo = SettingsRepository(app_root_dir=app_root)

    settings = repo.load_settings()

    assert settings["serial_template"]["fields"][-1]["key"] == "seq"
    backups = list(app_root.glob("settings.corrupt-*.json"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "{bad json"
    assert not settings_path.exists()
    shutil.rmtree(app_root, ignore_errors=True)


def test_invalid_settings_is_backed_up_and_defaulted() -> None:
    app_root = _case_root("invalid")
    settings_path = app_root / "settings.json"
    settings_path.write_text(
        json.dumps({"serial_template": {"fields": [{"type": "system", "key": "unknown"}]}}),
        encoding="utf-8",
    )
    repo = SettingsRepository(app_root_dir=app_root)

    settings = repo.load_settings()

    assert settings["serial_template"]["fields"][-1]["key"] == "seq"
    assert len(list(app_root.glob("settings.corrupt-*.json"))) == 1
    shutil.rmtree(app_root, ignore_errors=True)
