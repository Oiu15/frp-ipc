import json
import shutil
from pathlib import Path

from tests.fakes import FakeVar

from application.form_mapper import RecipeFormMapper
from core.models import Recipe, SectionPlanSnapshot, SectionTargetSnapshot
from repositories.recipe_repository import RecipeRepository
import pytest


class _FakeRecipeView:
    REQUIRED_VARS = (
        'recipe_name_var',
        'pipe_len_var',
        'clamp_var',
        'clamp_confirm_wait_s_var',
        'margin_h_var',
        'margin_t_var',
        'meas_total_len_var',
        'section_n_var',
        'teach_axes_mode_var',
        'od_std_var',
        'id_std_var',
        'od_tol_var',
        'section_sampling_mode_var',
        'points_per_rev_var',
        'min_cov_var',
        'sample_timeout_var',
        'max_revs_var',
        'sample_delay_s_var',
        'rot_vel_velmove_var',
        'fit_strategy_var',
        'od_use_edges_var',
        'id_use_fit_var',
        'id_single_enable_var',
        'id_single_k_var',
        'id_single_b_var',
        'split_scan_var',
        'disable_id_modbus_var',
        'split_keep_spinning_var',
        'split_slip_check_var',
        'calc_input_mode_var',
        'bin_count_var',
        'bin_method_var',
        'pp_mode_var',
        'theta_delay_s_var',
        'len_enable_var',
        'len_z_low_approach_var',
        'len_low_search_dist_var',
        'len_high_search_dist_var',
        'len_search_vel_var',
        'len_search_timeout_var',
        'len_tol_var',
        'len_high_margin_var',
        'len_debounce_k_var',
        'len_max_stale_ms_var',
        'len_backoff_var',
    )

    def __init__(self) -> None:
        self.recipe = Recipe()
        self._len_low_appr_legacy_z = None
        for name in self.REQUIRED_VARS:
            setattr(self, name, FakeVar())

    def get_var_value(self, name: str, default=None):
        var = getattr(self, name, None)
        if var is None:
            return default
        return var.get()

    def set_var_value(self, name: str, value) -> None:
        var = getattr(self, name, None)
        setter = getattr(var, "set", None)
        if not callable(setter):
            return
        try:
            setter(value)
        except Exception:
            pass

    def sync_combo_value(self, combo_name: str, value: str) -> None:
        return None

    def log_ax3_speed_trace(self, *args, **kwargs) -> None:
        return None

    def refresh_length_info(self) -> None:
        return None

    def set_len_low_approach_legacy_z(self, value) -> None:
        self._len_low_appr_legacy_z = value

    def after_recipe_data_applied(self) -> None:
        return None


class TestRecipeRepositoryCompat:
    def _case_root(self, name: str) -> Path:
        root = Path('.test-artifacts') / name
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        return root / 'FRP_IPC'

    def _fixture_file(self) -> Path:
        return Path(__file__).resolve().parent / 'fixtures' / 'recipe_compat' / 'legacy_recipe.json'

    def _install_legacy_sample(self, app_root: Path) -> None:
        recipes_dir = app_root / 'recipes'
        recipes_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._fixture_file(), recipes_dir / 'compat_recipe.json')
        (recipes_dir / 'index.json').write_text(json.dumps({'last_recipe': 'compat_recipe'}, ensure_ascii=False, indent=2), encoding='utf-8')

    def test_fixture_file_keeps_legacy_recipe_schema_shape(self) -> None:
        data = json.loads(self._fixture_file().read_text(encoding='utf-8'))
        for key in (
            'name',
            'pipe_len_mm',
            'meas_total_len_mm',
            'section_count',
            'scan_mode',
            'points_per_rev',
            'section_pos_z',
            'start_valid',
            'ax2_rot_abs',
        ):
            assert key in data

    def test_load_legacy_recipe_json_via_repository(self) -> None:
        app_root = self._case_root('recipe_repository_compat_load')
        self._install_legacy_sample(app_root)
        repo = RecipeRepository(root=app_root / 'recipes')

        assert 'compat_recipe' in repo.list_names()
        assert repo.load_index().get('last_recipe') == 'compat_recipe'

        data = repo.load('compat_recipe')
        assert data['name'] == 'compat_recipe'
        assert data['section_count'] == 3
        assert data['section_pos_z'] == [100.0, 800.0, 1500.0]
        assert data['ax2_rot_valid']

    def test_legacy_recipe_json_still_maps_to_current_recipe_model(self) -> None:
        app_root = self._case_root('recipe_repository_compat_mapper')
        self._install_legacy_sample(app_root)
        repo = RecipeRepository(root=app_root / 'recipes')
        data = repo.load('compat_recipe')

        host = _FakeRecipeView()
        mapper = RecipeFormMapper(host)
        mapper.apply_data_to_ui(data)
        recipe = host.recipe
        dumped = mapper.recipe_to_dict(recipe)

        assert recipe.name == 'compat_recipe'
        assert recipe.section_count == 3
        assert recipe.scan_mode == 'split'
        assert not recipe.id_single_enable
        assert recipe.id_single_k == pytest.approx(1.234)
        assert recipe.id_single_b == pytest.approx(-0.456)
        assert recipe.len_enable
        assert recipe.clamp_confirm_wait_s == pytest.approx(3.0)
        assert recipe.len_low_approach_abs == pytest.approx(111.1)
        assert list(recipe.section_pos_z) == [100.0, 800.0, 1500.0]
        assert recipe.section_plan is None
        assert recipe.start_valid
        assert recipe.start_ax0_abs == pytest.approx(40.0)
        assert recipe.ax2_rot_valid
        assert recipe.ax2_rot_abs == pytest.approx(60.0)

        assert dumped['name'] == 'compat_recipe'
        assert dumped['section_count'] == 3
        assert dumped['scan_mode'] == 'split'
        assert dumped['section_sampling_mode'] == 'split'
        assert dumped['sampling_window_mode'] == 'separate_channels'
        assert dumped['section_pos_z'] == [100.0, 800.0, 1500.0]
        assert float(dumped['clamp_confirm_wait_s']) == pytest.approx(3.0)
        assert dumped['ax2_len_valid']
        assert float(dumped['ax2_len_abs']) == pytest.approx(50.0)
        assert dumped['ax2_rot_valid']
        assert float(dumped['ax2_rot_abs']) == pytest.approx(60.0)

    def test_section_plan_round_trips_with_recipe_json_mapping(self) -> None:
        host = _FakeRecipeView()
        mapper = RecipeFormMapper(host)
        host.recipe = Recipe(
            name='with-section-plan',
            section_count=2,
            section_pos_z=[10.0, 20.0],
            section_plan=SectionPlanSnapshot(
                sections=[
                    SectionTargetSnapshot(1, 10.0, 13.0, 101.0, 201.0, 401.0, 'computed'),
                    SectionTargetSnapshot(2, 20.0, 23.0, 102.0, 202.0, 402.0, 'taught'),
                ]
            ),
        )

        dumped = mapper.recipe_to_dict(host.recipe)
        mapper.apply_data_to_ui(dumped)

        assert isinstance(host.recipe.section_plan, SectionPlanSnapshot)
        assert host.recipe.section_plan is not None
        assert host.recipe.section_pos_z == [10.0, 20.0]
        assert host.recipe.section_plan.sections[1].source == 'taught'
        assert mapper.recipe_to_dict(host.recipe)['section_plan']['sections'][1]['source'] == 'taught'

    def test_missing_planning_fields_do_not_inherit_previous_recipe_state(self) -> None:
        host = _FakeRecipeView()
        host.recipe.start_valid = True
        host.recipe.start_ax0_abs = 123.4
        host.recipe.standby_valid = True
        host.recipe.standby_ax0_abs = 11.0
        host.recipe.standby_ax1_abs = 22.0
        host.recipe.standby_ax4_abs = 33.0
        host.recipe.ax2_len_valid = True
        host.recipe.ax2_len_abs = 44.0
        host.recipe.ax2_rot_valid = True
        host.recipe.ax2_rot_abs = 55.0

        mapper = RecipeFormMapper(host)
        mapper.apply_data_to_ui(
            {
                'name': 'missing-planning',
                'pipe_len_mm': 1700.0,
                'clamp_occupy_mm': 300.0,
                'margin_head_mm': 20.0,
                'margin_tail_mm': 20.0,
                'section_count': 3,
                'points_per_rev': 90,
                'sample_coverage': 0.9,
                'section_timeout_s': 10.0,
                'max_revs': 2.0,
                'section_pos_z': [10.0, 20.0, 30.0],
            }
        )

        recipe = host.recipe
        assert not recipe.start_valid
        assert recipe.start_ax0_abs == 0.0
        assert not recipe.standby_valid
        assert recipe.standby_ax0_abs == 0.0
        assert recipe.standby_ax1_abs == 0.0
        assert recipe.standby_ax4_abs == 0.0
        assert not recipe.ax2_len_valid
        assert recipe.ax2_len_abs == 0.0
        assert not recipe.ax2_rot_valid
        assert recipe.ax2_rot_abs == 0.0

    def test_sampling_mode_round_trips_with_new_recipe_fields(self) -> None:
        host = _FakeRecipeView()
        mapper = RecipeFormMapper(host)

        mapper.apply_data_to_ui(
            {
                'name': 'sampling-spec',
                'pipe_len_mm': 1700.0,
                'clamp_occupy_mm': 300.0,
                'margin_head_mm': 20.0,
                'margin_tail_mm': 20.0,
                'section_count': 2,
                'section_sampling_mode': 'split',
                'sampling_window_mode': 'separate_channels',
                'points_per_rev': 180,
                'sample_coverage': 0.9,
                'section_timeout_s': 8.0,
                'max_revs': 3.0,
                'section_pos_z': [25.0, 50.0],
            }
        )

        recipe = mapper.ui_vars_to_recipe()
        dumped = mapper.recipe_to_dict(recipe)

        assert recipe.section_sampling_mode == 'split'
        assert recipe.sampling_window_mode == 'separate_channels'
        assert recipe.scan_mode == 'split'
        assert dumped['section_sampling_mode'] == 'split'
        assert dumped['sampling_window_mode'] == 'separate_channels'
        assert dumped['scan_mode'] == 'split'
        assert dumped['sample_delay_s'] == 0.0

    def test_sample_delay_round_trips_with_recipe_fields(self) -> None:
        host = _FakeRecipeView()
        mapper = RecipeFormMapper(host)

        mapper.apply_data_to_ui(
            {
                'name': 'sample-delay',
                'pipe_len_mm': 1700.0,
                'clamp_occupy_mm': 300.0,
                'margin_head_mm': 20.0,
                'margin_tail_mm': 20.0,
                'section_count': 2,
                'sample_delay_s': 1.25,
                'points_per_rev': 180,
                'sample_coverage': 0.9,
                'section_timeout_s': 8.0,
                'max_revs': 3.0,
                'section_pos_z': [25.0, 50.0],
            }
        )

        recipe = mapper.ui_vars_to_recipe()
        dumped = mapper.recipe_to_dict(recipe)

        assert recipe.sample_delay_s == pytest.approx(1.25)
        assert float(dumped['sample_delay_s']) == pytest.approx(1.25)
