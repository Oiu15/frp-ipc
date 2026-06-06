from __future__ import annotations


def test_flat_application_import_paths_remain_compatible() -> None:
    from application._host_axis_calibration import HostAxisCalibrationMixin
    from application._host_confirm import HostConfirmMixin
    from application._host_export import HostExportMixin
    from application._host_gauge_connection import HostGaugeConnectionMixin
    from application._host_identity import HostIdentityMixin
    from application._host_keytest import HostKeytestMixin
    from application._host_length_measurement import HostLengthMeasurementMixin
    from application._host_main_view import HostMainViewMixin
    from application._host_od_calibration import HostOdCalibrationMixin
    from application._host_recipe import HostRecipeMixin
    from application._host_teach import HostTeachMixin
    from application._host_ui import HostUIMixin
    from application._host_validation import HostValidationMixin
    from application.app_adapters import AppDeviceGateway, ScreenController, ScreenPresenter, ScreenUiContext
    from application.axis_calibration_state import AxisCalibrationState
    from application.plc_sync_reader import PlcSyncReader
    from application.recipe_form_mapper import RecipeFormMapper
    from application.ui_queue_adapters import WorkflowUiEventAdapter

    assert HostAxisCalibrationMixin.__name__ == "HostAxisCalibrationMixin"
    assert HostConfirmMixin.__name__ == "HostConfirmMixin"
    assert HostExportMixin.__name__ == "HostExportMixin"
    assert HostGaugeConnectionMixin.__name__ == "HostGaugeConnectionMixin"
    assert HostIdentityMixin.__name__ == "HostIdentityMixin"
    assert HostKeytestMixin.__name__ == "HostKeytestMixin"
    assert HostLengthMeasurementMixin.__name__ == "HostLengthMeasurementMixin"
    assert HostMainViewMixin.__name__ == "HostMainViewMixin"
    assert HostOdCalibrationMixin.__name__ == "HostOdCalibrationMixin"
    assert HostRecipeMixin.__name__ == "HostRecipeMixin"
    assert HostTeachMixin.__name__ == "HostTeachMixin"
    assert HostUIMixin.__name__ == "HostUIMixin"
    assert HostValidationMixin.__name__ == "HostValidationMixin"
    assert AppDeviceGateway.__name__ == "AppDeviceGateway"
    assert ScreenController.__name__ == "ScreenController"
    assert ScreenPresenter.__name__ == "ScreenPresenter"
    assert ScreenUiContext.__name__ == "ScreenUiContext"
    assert AxisCalibrationState.__name__ == "AxisCalibrationState"
    assert PlcSyncReader.__name__ == "PlcSyncReader"
    assert RecipeFormMapper.__name__ == "RecipeFormMapper"
    assert WorkflowUiEventAdapter.__name__ == "WorkflowUiEventAdapter"
