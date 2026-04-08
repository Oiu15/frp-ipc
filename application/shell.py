from __future__ import annotations

import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from application.ui_event_dispatcher import UiEventDispatcher
from core.recipe_store import RecipeStore
from drivers.gauge_driver import GaugeWorker
from drivers.plc_client import PlcWorker
from repositories.calibration_repository import CalibrationRepository


def _noop_ui_event_handler(_payload: Any) -> None:
    pass


@dataclass(slots=True)
class AppDependencies:
    ui_q: queue.Queue
    cmd_q: queue.Queue
    worker: PlcWorker
    gauge_worker: Optional[GaugeWorker]
    calibration_repository: CalibrationRepository
    recipe_store: RecipeStore
    device_ui_event_dispatcher: UiEventDispatcher
    measurement_ui_event_dispatcher: UiEventDispatcher


class ApplicationShell:
    """Thin application shell for lifecycle and dependency assembly."""

    DEVICE_UI_EVENTS = (
        "plc_ok",
        "plc_err",
        "plc_giveup",
        "plc_manual",
        "plc_read",
        "gauge_conn",
        "gauge_tx",
        "gauge_ok",
        "gauge_raw",
        "gauge_err",
    )

    MEASUREMENT_UI_EVENTS = (
        "op_confirm_show",
        "op_confirm_close",
        "auto_clear",
        "auto_len",
        "auto_progress",
        "auto_cov",
        "auto_straightness",
        "auto_postcalc",
        "auto_raw_points",
        "auto_row",
        "auto_state",
    )

    def __init__(self, app_root_dir: Path | None = None) -> None:
        self.app_root_dir = Path(app_root_dir) if app_root_dir is not None else self.default_app_root_dir()
        self.dependencies: AppDependencies | None = None
        self.app: Any | None = None

    @staticmethod
    def default_app_root_dir() -> Path:
        try:
            return Path.home() / "FRP_IPC"
        except Exception:
            return Path("./FRP_IPC")

    @classmethod
    def _build_ui_event_dispatcher(cls, event_names: tuple[str, ...]) -> UiEventDispatcher:
        return UiEventDispatcher({event_name: _noop_ui_event_handler for event_name in event_names})

    @classmethod
    def build_device_ui_event_dispatcher(cls) -> UiEventDispatcher:
        return cls._build_ui_event_dispatcher(cls.DEVICE_UI_EVENTS)

    @classmethod
    def build_measurement_ui_event_dispatcher(cls) -> UiEventDispatcher:
        return cls._build_ui_event_dispatcher(cls.MEASUREMENT_UI_EVENTS)

    def assemble_dependencies(self) -> AppDependencies:
        if self.dependencies is not None:
            return self.dependencies

        ui_q: queue.Queue = queue.Queue()
        cmd_q: queue.Queue = queue.Queue()

        worker = PlcWorker(ui_q, cmd_q)
        worker.start()

        gauge_worker: Optional[GaugeWorker] = GaugeWorker(ui_q)
        gauge_worker.start()

        calibration_repository = CalibrationRepository(app_root_dir=self.app_root_dir)

        try:
            recipe_store = RecipeStore(RecipeStore.default_root("FRP_IPC"))
        except Exception:
            recipe_store = RecipeStore(Path("./data/recipes"))

        self.dependencies = AppDependencies(
            ui_q=ui_q,
            cmd_q=cmd_q,
            worker=worker,
            gauge_worker=gauge_worker,
            calibration_repository=calibration_repository,
            recipe_store=recipe_store,
            device_ui_event_dispatcher=self.build_device_ui_event_dispatcher(),
            measurement_ui_event_dispatcher=self.build_measurement_ui_event_dispatcher(),
        )
        return self.dependencies

    def create_app(self, app_factory: Callable[..., Any]) -> Any:
        dependencies = self.assemble_dependencies()
        try:
            app = app_factory(dependencies=dependencies, shell=self)
        except Exception:
            self.stop_workers()
            raise
        self.app = app
        return app

    def run(self, app_factory: Callable[..., Any]) -> Any:
        app = self.create_app(app_factory)
        try:
            app.mainloop()
        finally:
            self.app = None
            try:
                if app.winfo_exists():
                    self.close_app(app)
            except Exception:
                pass
        return app

    def stop_workers(self) -> None:
        dependencies = self.dependencies
        if dependencies is None:
            return
        try:
            dependencies.worker.stop()
        except Exception:
            pass
        try:
            if dependencies.gauge_worker is not None:
                dependencies.gauge_worker.stop()
        except Exception:
            pass

    def close_app(self, app: Any) -> None:
        try:
            auto_thread = getattr(app, "_auto_thread", None)
            if auto_thread is not None and auto_thread.is_alive():
                auto_thread.stop()
        except Exception:
            pass
        self.stop_workers()
        try:
            app.destroy()
        except Exception:
            pass
        if self.app is app:
            self.app = None


__all__ = ["AppDependencies", "ApplicationShell"]
