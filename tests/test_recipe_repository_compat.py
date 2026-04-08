import json
import shutil
import unittest
from pathlib import Path

from application.recipe_form_mapper import RecipeFormMapper
from core.models import Recipe
from repositories.recipe_repository import RecipeRepository


class _FakeVar:
    def __init__(self, value=None) -> None:
        self._value = value

    def get(self):
        return self._value

    def set(self, value) -> None:
        self._value = value


class _FakeHost:
    REQUIRED_VARS = (
        'recipe_name_var',
        'pipe_len_var',
        'clamp_var',
        'margin_h_var',
        'margin_t_var',
        'meas_total_len_var',
        'section_n_var',
        'teach_axes_mode_var',
        'od_std_var',
        'id_std_var',
        'od_tol_var',
        'points_per_rev_var',
        'min_cov_var',
        'sample_timeout_var',
        'max_revs_var',
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
        for name in self.REQUIRED_VARS:
            setattr(self, name, _FakeVar())

    def _log_ax3_speed_trace(self, *args, **kwargs) -> None:
        return None


class RecipeRepositoryCompatTest(unittest.TestCase):
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
            self.assertIn(key, data)

    def test_load_legacy_recipe_json_via_repository(self) -> None:
        app_root = self._case_root('recipe_repository_compat_load')
        self._install_legacy_sample(app_root)
        repo = RecipeRepository(root=app_root / 'recipes')

        self.assertIn('compat_recipe', repo.list_names())
        self.assertEqual(repo.load_index().get('last_recipe'), 'compat_recipe')

        data = repo.load('compat_recipe')
        self.assertEqual(data['name'], 'compat_recipe')
        self.assertEqual(data['section_count'], 3)
        self.assertEqual(data['section_pos_z'], [100.0, 800.0, 1500.0])
        self.assertTrue(data['ax2_rot_valid'])

    def test_legacy_recipe_json_still_maps_to_current_recipe_model(self) -> None:
        app_root = self._case_root('recipe_repository_compat_mapper')
        self._install_legacy_sample(app_root)
        repo = RecipeRepository(root=app_root / 'recipes')
        data = repo.load('compat_recipe')

        host = _FakeHost()
        mapper = RecipeFormMapper(host)
        mapper.apply_data_to_ui(data)
        recipe = host.recipe
        dumped = mapper.recipe_to_dict(recipe)

        self.assertEqual(recipe.name, 'compat_recipe')
        self.assertEqual(recipe.section_count, 3)
        self.assertEqual(recipe.scan_mode, 'split')
        self.assertTrue(recipe.id_single_enable)
        self.assertAlmostEqual(recipe.id_single_k, 1.234, places=6)
        self.assertAlmostEqual(recipe.id_single_b, -0.456, places=6)
        self.assertTrue(recipe.len_enable)
        self.assertAlmostEqual(recipe.len_low_approach_abs, 111.1, places=6)
        self.assertEqual(list(recipe.section_pos_z), [100.0, 800.0, 1500.0])
        self.assertTrue(recipe.start_valid)
        self.assertAlmostEqual(recipe.start_ax0_abs, 40.0, places=6)
        self.assertTrue(recipe.ax2_rot_valid)
        self.assertAlmostEqual(recipe.ax2_rot_abs, 60.0, places=6)

        self.assertEqual(dumped['name'], 'compat_recipe')
        self.assertEqual(dumped['section_count'], 3)
        self.assertEqual(dumped['scan_mode'], 'split')
        self.assertEqual(dumped['section_pos_z'], [100.0, 800.0, 1500.0])
        self.assertTrue(dumped['ax2_len_valid'])
        self.assertAlmostEqual(float(dumped['ax2_len_abs']), 50.0, places=6)
        self.assertTrue(dumped['ax2_rot_valid'])
        self.assertAlmostEqual(float(dumped['ax2_rot_abs']), 60.0, places=6)


if __name__ == '__main__':
    unittest.main()