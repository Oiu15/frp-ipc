from __future__ import annotations

"""Recipe store: manage recipes saved as JSON in user's home directory.

- One recipe per file: <name>.json
- Index file: index.json (stores last used recipe)
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

_INVALID_FS_CHARS = r'<>:"/\\|?*'


def sanitize_recipe_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise ValueError("配方名不能为空")
    name = re.sub(f"[{re.escape(_INVALID_FS_CHARS)}]", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    if not name:
        raise ValueError("配方名非法")
    # avoid reserved names on Windows (CON, PRN, etc.) by appending underscore
    if name.upper() in {"CON","PRN","AUX","NUL","COM1","COM2","COM3","COM4","COM5","COM6","COM7","COM8","COM9",
                        "LPT1","LPT2","LPT3","LPT4","LPT5","LPT6","LPT7","LPT8","LPT9"}:
        name = name + "_"
    return name


@dataclass
class RecipeStore:
    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def default_root(app_dir_name: str = "FRP_IPC") -> Path:
        # User directory per requirement: put under user's home.
        # Example: C:\Users\<user>\FRP_IPC\recipes
        return Path.home() / app_dir_name / "recipes"

    def _path_of(self, name: str) -> Path:
        safe = sanitize_recipe_name(name)
        return self.root / f"{safe}.json"

    def list_names(self) -> List[str]:
        names: List[str] = []
        for p in self.root.glob("*.json"):
            if p.name.lower() == "index.json":
                continue
            names.append(p.stem)
        names.sort(key=lambda s: s.lower())
        return names

    def load(self, name: str) -> Dict[str, Any]:
        p = self._path_of(name)
        if not p.exists():
            raise FileNotFoundError(f"配方不存在：{name}")
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, name: str, data: Dict[str, Any]) -> str:
        safe = sanitize_recipe_name(name)
        p = self._path_of(safe)
        tmp = p.with_suffix(p.suffix + ".tmp")
        payload = dict(data)
        payload.setdefault("_meta", {})
        try:
            payload["_meta"].update({"name": safe, "schema": "frp_recipe_v1"})
        except Exception:
            payload["_meta"] = {"name": safe, "schema": "frp_recipe_v1"}
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp.replace(p)
        return safe

    def delete(self, name: str) -> None:
        p = self._path_of(name)
        if p.exists():
            p.unlink()

    def load_index(self) -> Dict[str, Any]:
        p = self.root / "index.json"
        if not p.exists():
            return {}
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    def save_index(self, index: Dict[str, Any]) -> None:
        p = self.root / "index.json"
        tmp = p.with_suffix(p.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        tmp.replace(p)
