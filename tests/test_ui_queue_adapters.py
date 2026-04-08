import queue
import unittest

from application.ui_queue_adapters import WorkerUiEventAdapter, WorkflowUiEventAdapter
from core.models import AxisComm, MeasureRow


class UiQueueAdaptersTest(unittest.TestCase):
    def test_worker_adapter_preserves_legacy_plc_payload_shape(self) -> None:
        ui_q: queue.Queue = queue.Queue()
        adapter = WorkerUiEventAdapter(ui_q)
        axes = [AxisComm(act_pos=12.5)]

        adapter.publish_plc_ok(
            axes=axes,
            cl_out4_mm=152.7,
            cl_out4_cnt=3,
            keytest_x_bits=[1, 0, 1],
            keytest_y_bits=[0, 1],
        )

        event = ui_q.get_nowait()
        self.assertEqual(event[0], 'plc_ok')
        payload = event[1]
        self.assertEqual(payload['axes'], axes)
        self.assertEqual(payload['cl_out4_mm'], 152.7)
        self.assertEqual(payload['cl_out4_cnt'], 3)
        self.assertEqual(payload['keytest_x_bits'], [1, 0, 1])
        self.assertEqual(payload['keytest_y_bits'], [0, 1])

    def test_worker_adapter_preserves_legacy_gauge_payload_shape(self) -> None:
        ui_q: queue.Queue = queue.Queue()
        adapter = WorkerUiEventAdapter(ui_q)

        adapter.publish_gauge_ok(
            ts=123.4,
            od=187.3,
            judge='GO',
            od2=187.1,
            judge2='GO',
            raw='M0,1,+187.3,GO,+187.1,GO',
        )

        self.assertEqual(
            ui_q.get_nowait(),
            (
                'gauge_ok',
                {
                    'ts': 123.4,
                    'od': 187.3,
                    'judge': 'GO',
                    'od2': 187.1,
                    'judge2': 'GO',
                    'raw': 'M0,1,+187.3,GO,+187.1,GO',
                },
            ),
        )

    def test_workflow_adapter_preserves_legacy_progress_payload_shape(self) -> None:
        ui_q: queue.Queue = queue.Queue()
        adapter = WorkflowUiEventAdapter(ui_q)

        adapter.publish_progress(section_index=2, section_total=5, z_pos_mm=100.0, ax0_abs=200.0)

        self.assertEqual(
            ui_q.get_nowait(),
            (
                'auto_progress',
                {'idx': 1, 'total': 5, 'x_ui': 100.0, 'x_abs': 200.0},
            ),
        )

    def test_workflow_adapter_preserves_legacy_row_and_raw_points_shape(self) -> None:
        ui_q: queue.Queue = queue.Queue()
        adapter = WorkflowUiEventAdapter(ui_q)
        row = MeasureRow(
            idx=1,
            x_ui=100.0,
            x_abs=200.0,
            od_avg=187.31,
            od_dev=0.01,
            od_runout=0.02,
            od_round=0.03,
            id_avg=152.70,
            id_dev=0.01,
            id_runout=0.02,
            id_round=0.03,
            concentricity=0.04,
        )

        adapter.publish_row(row)
        adapter.publish_raw_points([{'section_idx': 1, 'theta_deg': 0.0, 'od_mm': 187.3}])

        self.assertEqual(ui_q.get_nowait(), ('auto_row', {'row': row}))
        self.assertEqual(
            ui_q.get_nowait(),
            ('auto_raw_points', {'points': [{'section_idx': 1, 'theta_deg': 0.0, 'od_mm': 187.3}]}),
        )


if __name__ == '__main__':
    unittest.main()
