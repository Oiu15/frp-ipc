from __future__ import annotations

"""Thin repository wrapper over the existing core.recipe_store module.

Current goal:
- keep the storage backend unchanged (`RecipeStore`)
- let upper layers depend on `repositories.recipe_repository` instead of
  reaching into `core/` directly
"""

from pathlib import Path
from typing import Any

from core.recipe_store import RecipeStore, sanitize_recipe_name


class RecipeRepository:
    """Repository facade backed by the legacy RecipeStore."""

    def __init__(self, root: Path | None = None, *, store: RecipeStore | None = None) -> None:
        if store is not None:
            self._store = store
        else:
            resolved_root = root if root is not None else self.default_root("FRP_IPC")
            self._store = RecipeStore(Path(resolved_root))

    @property
    def root(self) -> Path:
        return self._store.root

    @staticmethod
    def default_root(app_dir_name: str = "FRP_IPC") -> Path:
        return RecipeStore.default_root(app_dir_name)

    @staticmethod
    def sanitize_name(name: str) -> str:
        return sanitize_recipe_name(name)

    def list_names(self) -> list[str]:
        return self._store.list_names()

    def load(self, name: str) -> dict[str, Any]:
        return self._store.load(name)

    def save(self, name: str, data: dict[str, Any]) -> str:
        return self._store.save(name, data)

    def delete(self, name: str) -> None:
        self._store.delete(name)

    def load_index(self) -> dict[str, Any]:
        return self._store.load_index()

    def save_index(self, index: dict[str, Any]) -> None:
        self._store.save_index(index)

    def path_of(self, name: str) -> Path:
        safe = self.sanitize_name(name)
        return self.root / f"{safe}.json"


__all__ = ["RecipeRepository"]
