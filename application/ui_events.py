from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, TypeAlias, cast

from core.models import AxisComm, MeasureRow

UiEventTuple: TypeAlias = tuple[str, Any]


def _as_mapping(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        return payload
    return {}


def _copy_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        return dict(payload)
    return {}


def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _to_bool(value: Any) -> bool | None:
    if value is None:
        return None
    try:
        return bool(value)
    except Exception:
        return None


class UiEventBase:
    event_name: ClassVar[str]

    @classmethod
    def from_payload(cls, payload: Any):
        raise NotImplementedError

    def to_payload(self) -> dict[str, Any]:
        raise NotImplementedError

    def to_ui_event(self) -> UiEventTuple:
        return (self.event_name, self.to_payload())


@dataclass(slots=True)
class PlcOkEvent(UiEventBase):
    event_name: ClassVar[str] = 'plc_ok'

    axes: list[AxisComm]
    cl_out1_raw: int | None = None
    cl_out1_mm: float | None = None
    cl_out1_cnt: int | None = None
    cl_out2_raw: int | None = None
    cl_out2_mm: float | None = None
    cl_out2_cnt: int | None = None
    cl_out3_raw: int | None = None
    cl_out3_mm: float | None = None
    cl_out3_cnt: int | None = None
    cl_out4_raw: int | None = None
    cl_out4_mm: float | None = None
    cl_out4_cnt: int | None = None
    cl_out5_raw: int | None = None
    cl_out5_mm: float | None = None
    cl_out5_cnt: int | None = None
    keytest_x_bits: list[int] | None = None
    keytest_y_bits: list[int] | None = None

    @classmethod
    def from_payload(cls, payload: Any) -> 'PlcOkEvent':
        data = _as_mapping(payload)
        axes = list(cast(list[AxisComm], data.get('axes', []) or []))
        return cls(
            axes=axes,
            cl_out1_raw=_to_int(data.get('cl_out1_raw')),
            cl_out1_mm=_to_float(data.get('cl_out1_mm')),
            cl_out1_cnt=_to_int(data.get('cl_out1_cnt')),
            cl_out2_raw=_to_int(data.get('cl_out2_raw')),
            cl_out2_mm=_to_float(data.get('cl_out2_mm')),
            cl_out2_cnt=_to_int(data.get('cl_out2_cnt')),
            cl_out3_raw=_to_int(data.get('cl_out3_raw')),
            cl_out3_mm=_to_float(data.get('cl_out3_mm')),
            cl_out3_cnt=_to_int(data.get('cl_out3_cnt')),
            cl_out4_raw=_to_int(data.get('cl_out4_raw')),
            cl_out4_mm=_to_float(data.get('cl_out4_mm')),
            cl_out4_cnt=_to_int(data.get('cl_out4_cnt')),
            cl_out5_raw=_to_int(data.get('cl_out5_raw')),
            cl_out5_mm=_to_float(data.get('cl_out5_mm')),
            cl_out5_cnt=_to_int(data.get('cl_out5_cnt')),
            keytest_x_bits=list(data.get('keytest_x_bits', []) or []) or None,
            keytest_y_bits=list(data.get('keytest_y_bits', []) or []) or None,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            'axes': list(self.axes),
            'cl_out1_raw': self.cl_out1_raw,
            'cl_out1_mm': self.cl_out1_mm,
            'cl_out1_cnt': self.cl_out1_cnt,
            'cl_out2_raw': self.cl_out2_raw,
            'cl_out2_mm': self.cl_out2_mm,
            'cl_out2_cnt': self.cl_out2_cnt,
            'cl_out3_raw': self.cl_out3_raw,
            'cl_out3_mm': self.cl_out3_mm,
            'cl_out3_cnt': self.cl_out3_cnt,
            'cl_out4_raw': self.cl_out4_raw,
            'cl_out4_mm': self.cl_out4_mm,
            'cl_out4_cnt': self.cl_out4_cnt,
            'cl_out5_raw': self.cl_out5_raw,
            'cl_out5_mm': self.cl_out5_mm,
            'cl_out5_cnt': self.cl_out5_cnt,
            'keytest_x_bits': list(self.keytest_x_bits or []),
            'keytest_y_bits': list(self.keytest_y_bits or []),
        }


@dataclass(slots=True)
class PlcErrEvent(UiEventBase):
    event_name: ClassVar[str] = 'plc_err'

    err: str
    retry: int | None = None
    max: int | None = None
    backoff_s: float | None = None

    @classmethod
    def from_payload(cls, payload: Any) -> 'PlcErrEvent':
        data = _as_mapping(payload)
        return cls(
            err=str(data.get('err', '') or ''),
            retry=_to_int(data.get('retry')),
            max=_to_int(data.get('max')),
            backoff_s=_to_float(data.get('backoff_s')),
        )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {'err': self.err}
        if self.retry is not None:
            payload['retry'] = self.retry
        if self.max is not None:
            payload['max'] = self.max
        if self.backoff_s is not None:
            payload['backoff_s'] = self.backoff_s
        return payload


@dataclass(slots=True)
class PlcGiveupEvent(UiEventBase):
    event_name: ClassVar[str] = 'plc_giveup'

    retry: int
    max: int

    @classmethod
    def from_payload(cls, payload: Any) -> 'PlcGiveupEvent':
        data = _as_mapping(payload)
        return cls(retry=int(data.get('retry', 0) or 0), max=int(data.get('max', 0) or 0))

    def to_payload(self) -> dict[str, Any]:
        return {'retry': self.retry, 'max': self.max}


@dataclass(slots=True)
class PlcManualEvent(UiEventBase):
    event_name: ClassVar[str] = 'plc_manual'

    ip: str
    port: int

    @classmethod
    def from_payload(cls, payload: Any) -> 'PlcManualEvent':
        data = _as_mapping(payload)
        return cls(ip=str(data.get('ip', '') or ''), port=int(data.get('port', 0) or 0))

    def to_payload(self) -> dict[str, Any]:
        return {'ip': self.ip, 'port': self.port}


@dataclass(slots=True)
class PlcReadEvent(UiEventBase):
    event_name: ClassVar[str] = 'plc_read'

    tag: str
    d_addr: int
    count: int
    regs: list[int]
    t_uiq_put_ns: int | None = None

    @classmethod
    def from_payload(cls, payload: Any) -> 'PlcReadEvent':
        data = _as_mapping(payload)
        return cls(
            tag=str(data.get('tag', '') or ''),
            d_addr=int(data.get('d_addr', 0) or 0),
            count=int(data.get('count', 0) or 0),
            regs=[int(v) for v in list(data.get('regs', []) or [])],
            t_uiq_put_ns=_to_int(data.get('t_uiq_put_ns')),
        )

    def to_payload(self) -> dict[str, Any]:
        payload = {'tag': self.tag, 'd_addr': self.d_addr, 'count': self.count, 'regs': list(self.regs)}
        if self.t_uiq_put_ns is not None:
            payload['t_uiq_put_ns'] = self.t_uiq_put_ns
        return payload


@dataclass(slots=True)
class GaugeConnEvent(UiEventBase):
    event_name: ClassVar[str] = 'gauge_conn'

    ts: float
    connected: bool
    port: str | None = None
    baud: int | None = None

    @classmethod
    def from_payload(cls, payload: Any) -> 'GaugeConnEvent':
        data = _as_mapping(payload)
        return cls(
            ts=float(data.get('ts', 0.0) or 0.0),
            connected=bool(data.get('connected', False)),
            port=(str(data.get('port')) if data.get('port') is not None else None),
            baud=_to_int(data.get('baud')),
        )

    def to_payload(self) -> dict[str, Any]:
        payload = {'ts': self.ts, 'connected': self.connected}
        if self.port is not None:
            payload['port'] = self.port
        if self.baud is not None:
            payload['baud'] = self.baud
        return payload


@dataclass(slots=True)
class GaugeTxEvent(UiEventBase):
    event_name: ClassVar[str] = 'gauge_tx'

    ts: float
    cmd: str

    @classmethod
    def from_payload(cls, payload: Any) -> 'GaugeTxEvent':
        data = _as_mapping(payload)
        return cls(ts=float(data.get('ts', 0.0) or 0.0), cmd=str(data.get('cmd', '') or ''))

    def to_payload(self) -> dict[str, Any]:
        return {'ts': self.ts, 'cmd': self.cmd}


@dataclass(slots=True)
class GaugeRawEvent(UiEventBase):
    event_name: ClassVar[str] = 'gauge_raw'

    ts: float
    raw: str

    @classmethod
    def from_payload(cls, payload: Any) -> 'GaugeRawEvent':
        data = _as_mapping(payload)
        return cls(ts=float(data.get('ts', 0.0) or 0.0), raw=str(data.get('raw', '') or ''))

    def to_payload(self) -> dict[str, Any]:
        return {'ts': self.ts, 'raw': self.raw}


@dataclass(slots=True)
class GaugeErrEvent(UiEventBase):
    event_name: ClassVar[str] = 'gauge_err'

    ts: float
    err: str

    @classmethod
    def from_payload(cls, payload: Any) -> 'GaugeErrEvent':
        data = _as_mapping(payload)
        return cls(ts=float(data.get('ts', 0.0) or 0.0), err=str(data.get('err', '') or ''))

    def to_payload(self) -> dict[str, Any]:
        return {'ts': self.ts, 'err': self.err}


@dataclass(slots=True)
class GaugeOkEvent(UiEventBase):
    event_name: ClassVar[str] = 'gauge_ok'

    ts: float
    od: float
    judge: str
    od2: float | None = None
    judge2: str = 'UNK'
    raw: str = ''

    @classmethod
    def from_payload(cls, payload: Any) -> 'GaugeOkEvent':
        data = _as_mapping(payload)
        return cls(
            ts=float(data.get('ts', 0.0) or 0.0),
            od=float(data.get('od', 0.0) or 0.0),
            judge=str(data.get('judge', 'UNK') or 'UNK'),
            od2=_to_float(data.get('od2')),
            judge2=str(data.get('judge2', 'UNK') or 'UNK'),
            raw=str(data.get('raw', '') or ''),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            'ts': self.ts,
            'od': self.od,
            'judge': self.judge,
            'od2': self.od2,
            'judge2': self.judge2,
            'raw': self.raw,
        }


@dataclass(slots=True)
class OpConfirmShowEvent(UiEventBase):
    event_name: ClassVar[str] = 'op_confirm_show'

    token: Any
    title: str
    message: str
    allow_stop: bool = False

    @classmethod
    def from_payload(cls, payload: Any) -> 'OpConfirmShowEvent':
        data = _as_mapping(payload)
        return cls(
            token=data.get('token'),
            title=str(data.get('title', '') or ''),
            message=str(data.get('message', '') or ''),
            allow_stop=bool(data.get('allow_stop', False)),
        )

    def to_payload(self) -> dict[str, Any]:
        return {'token': self.token, 'title': self.title, 'message': self.message, 'allow_stop': self.allow_stop}


@dataclass(slots=True)
class OpConfirmCloseEvent(UiEventBase):
    event_name: ClassVar[str] = 'op_confirm_close'

    token: Any

    @classmethod
    def from_payload(cls, payload: Any) -> 'OpConfirmCloseEvent':
        data = _as_mapping(payload)
        return cls(token=data.get('token'))

    def to_payload(self) -> dict[str, Any]:
        return {'token': self.token}


@dataclass(slots=True)
class AutoStateEvent(UiEventBase):
    event_name: ClassVar[str] = 'auto_state'

    state: str
    msg: str

    @classmethod
    def from_payload(cls, payload: Any) -> 'AutoStateEvent':
        data = _as_mapping(payload)
        return cls(state=str(data.get('state', 'IDLE') or 'IDLE'), msg=str(data.get('msg', '-') or '-'))

    def to_payload(self) -> dict[str, Any]:
        return {'state': self.state, 'msg': self.msg}


@dataclass(slots=True)
class AutoProgressEvent(UiEventBase):
    event_name: ClassVar[str] = 'auto_progress'

    idx: int
    total: int
    x_ui: float
    x_abs: float

    @classmethod
    def from_payload(cls, payload: Any) -> 'AutoProgressEvent':
        data = _as_mapping(payload)
        return cls(
            idx=int(data.get('idx', 0) or 0),
            total=int(data.get('total', 0) or 0),
            x_ui=float(data.get('x_ui', 0.0) or 0.0),
            x_abs=float(data.get('x_abs', 0.0) or 0.0),
        )

    def to_payload(self) -> dict[str, Any]:
        return {'idx': self.idx, 'total': self.total, 'x_ui': self.x_ui, 'x_abs': self.x_abs}


@dataclass(slots=True)
class AutoLenEvent(UiEventBase):
    event_name: ClassVar[str] = 'auto_len'

    data: dict[str, Any]

    @classmethod
    def from_payload(cls, payload: Any) -> 'AutoLenEvent':
        return cls(data=_copy_dict(payload))

    def to_payload(self) -> dict[str, Any]:
        return dict(self.data)


@dataclass(slots=True)
class AutoCoverageEvent(UiEventBase):
    event_name: ClassVar[str] = 'auto_cov'

    idx: int | None = None
    cov: float | None = None
    miss: int | None = None
    max_gap_deg: float | None = None
    reason: str = ''
    revs: float | None = None
    elapsed: float | None = None
    split_shift_deg: float | None = None
    coax_unreliable: bool | None = None
    keep_spinning: bool | None = None

    @classmethod
    def from_payload(cls, payload: Any) -> 'AutoCoverageEvent':
        data = _as_mapping(payload)
        return cls(
            idx=_to_int(data.get('idx')),
            cov=_to_float(data.get('cov')),
            miss=_to_int(data.get('miss')),
            max_gap_deg=_to_float(data.get('max_gap_deg')),
            reason=str(data.get('reason', '') or ''),
            revs=_to_float(data.get('revs')),
            elapsed=_to_float(data.get('elapsed')),
            split_shift_deg=_to_float(data.get('split_shift_deg')),
            coax_unreliable=_to_bool(data.get('coax_unreliable')),
            keep_spinning=_to_bool(data.get('keep_spinning')),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            'idx': self.idx,
            'cov': self.cov,
            'miss': self.miss,
            'max_gap_deg': self.max_gap_deg,
            'reason': self.reason,
            'revs': self.revs,
            'elapsed': self.elapsed,
            'split_shift_deg': self.split_shift_deg,
            'coax_unreliable': self.coax_unreliable,
            'keep_spinning': self.keep_spinning,
        }


@dataclass(slots=True)
class AutoRawPointsEvent(UiEventBase):
    event_name: ClassVar[str] = 'auto_raw_points'

    points: list[dict[str, Any]]

    @classmethod
    def from_payload(cls, payload: Any) -> 'AutoRawPointsEvent':
        data = _as_mapping(payload)
        points = [dict(p) for p in list(data.get('points', []) or []) if isinstance(p, Mapping)]
        return cls(points=points)

    def to_payload(self) -> dict[str, Any]:
        return {'points': [dict(p) for p in self.points]}


@dataclass(slots=True)
class AutoRowEvent(UiEventBase):
    event_name: ClassVar[str] = 'auto_row'

    row: MeasureRow

    @classmethod
    def from_payload(cls, payload: Any) -> 'AutoRowEvent':
        data = _as_mapping(payload)
        return cls(row=cast(MeasureRow, data.get('row')))

    def to_payload(self) -> dict[str, Any]:
        return {'row': self.row}


@dataclass(slots=True)
class AutoStraightnessEvent(UiEventBase):
    event_name: ClassVar[str] = 'auto_straightness'

    straight_od: float | None = None
    straight_id: float | None = None
    axis_dist: float | None = None
    conc_max: float | None = None
    axis_span_max: float | None = None
    od_tilt_deg: float | None = None
    od_end_off_mm: float | None = None
    od_slope: float | None = None
    id_tilt_deg: float | None = None
    id_end_off_mm: float | None = None
    id_slope: float | None = None

    @classmethod
    def from_payload(cls, payload: Any) -> 'AutoStraightnessEvent':
        data = _as_mapping(payload)
        return cls(
            straight_od=_to_float(data.get('straight_od')),
            straight_id=_to_float(data.get('straight_id')),
            axis_dist=_to_float(data.get('axis_dist')),
            conc_max=_to_float(data.get('conc_max')),
            axis_span_max=_to_float(data.get('axis_span_max')),
            od_tilt_deg=_to_float(data.get('od_tilt_deg')),
            od_end_off_mm=_to_float(data.get('od_end_off_mm')),
            od_slope=_to_float(data.get('od_slope')),
            id_tilt_deg=_to_float(data.get('id_tilt_deg')),
            id_end_off_mm=_to_float(data.get('id_end_off_mm')),
            id_slope=_to_float(data.get('id_slope')),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            'straight_od': self.straight_od,
            'straight_id': self.straight_id,
            'axis_dist': self.axis_dist,
            'conc_max': self.conc_max,
            'axis_span_max': self.axis_span_max,
            'od_tilt_deg': self.od_tilt_deg,
            'od_end_off_mm': self.od_end_off_mm,
            'od_slope': self.od_slope,
            'id_tilt_deg': self.id_tilt_deg,
            'id_end_off_mm': self.id_end_off_mm,
            'id_slope': self.id_slope,
        }


@dataclass(slots=True)
class AutoPostcalcEvent(UiEventBase):
    event_name: ClassVar[str] = 'auto_postcalc'

    ecc_od: list[Any] | None = None
    ecc_id: list[Any] | None = None
    straight_od: float | None = None
    straight_id: float | None = None
    axis_dist: float | None = None
    conc_max: float | None = None
    axis_span_max: float | None = None
    od_tilt_deg: float | None = None
    od_end_off_mm: float | None = None
    od_slope: float | None = None
    id_tilt_deg: float | None = None
    id_end_off_mm: float | None = None
    id_slope: float | None = None

    @classmethod
    def from_payload(cls, payload: Any) -> 'AutoPostcalcEvent':
        data = _as_mapping(payload)
        return cls(
            ecc_od=list(data.get('ecc_od', []) or []) or None,
            ecc_id=list(data.get('ecc_id', []) or []) or None,
            straight_od=_to_float(data.get('straight_od')),
            straight_id=_to_float(data.get('straight_id')),
            axis_dist=_to_float(data.get('axis_dist')),
            conc_max=_to_float(data.get('conc_max')),
            axis_span_max=_to_float(data.get('axis_span_max')),
            od_tilt_deg=_to_float(data.get('od_tilt_deg')),
            od_end_off_mm=_to_float(data.get('od_end_off_mm')),
            od_slope=_to_float(data.get('od_slope')),
            id_tilt_deg=_to_float(data.get('id_tilt_deg')),
            id_end_off_mm=_to_float(data.get('id_end_off_mm')),
            id_slope=_to_float(data.get('id_slope')),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            'ecc_od': list(self.ecc_od or []),
            'ecc_id': list(self.ecc_id or []),
            'straight_od': self.straight_od,
            'straight_id': self.straight_id,
            'axis_dist': self.axis_dist,
            'conc_max': self.conc_max,
            'axis_span_max': self.axis_span_max,
            'od_tilt_deg': self.od_tilt_deg,
            'od_end_off_mm': self.od_end_off_mm,
            'od_slope': self.od_slope,
            'id_tilt_deg': self.id_tilt_deg,
            'id_end_off_mm': self.id_end_off_mm,
            'id_slope': self.id_slope,
        }


@dataclass(slots=True)
class AutoClearEvent(UiEventBase):
    event_name: ClassVar[str] = 'auto_clear'

    ts: float | None = None

    @classmethod
    def from_payload(cls, payload: Any) -> 'AutoClearEvent':
        data = _as_mapping(payload)
        return cls(ts=_to_float(data.get('ts')))

    def to_payload(self) -> dict[str, Any]:
        return {'ts': self.ts}


KnownUiEvent: TypeAlias = (
    PlcOkEvent
    | PlcErrEvent
    | PlcGiveupEvent
    | PlcManualEvent
    | PlcReadEvent
    | GaugeConnEvent
    | GaugeTxEvent
    | GaugeRawEvent
    | GaugeErrEvent
    | GaugeOkEvent
    | OpConfirmShowEvent
    | OpConfirmCloseEvent
    | AutoStateEvent
    | AutoProgressEvent
    | AutoLenEvent
    | AutoCoverageEvent
    | AutoRawPointsEvent
    | AutoRowEvent
    | AutoStraightnessEvent
    | AutoPostcalcEvent
    | AutoClearEvent
)


UI_EVENT_TYPES: dict[str, type[UiEventBase]] = {
    PlcOkEvent.event_name: PlcOkEvent,
    PlcErrEvent.event_name: PlcErrEvent,
    PlcGiveupEvent.event_name: PlcGiveupEvent,
    PlcManualEvent.event_name: PlcManualEvent,
    PlcReadEvent.event_name: PlcReadEvent,
    GaugeConnEvent.event_name: GaugeConnEvent,
    GaugeTxEvent.event_name: GaugeTxEvent,
    GaugeRawEvent.event_name: GaugeRawEvent,
    GaugeErrEvent.event_name: GaugeErrEvent,
    GaugeOkEvent.event_name: GaugeOkEvent,
    OpConfirmShowEvent.event_name: OpConfirmShowEvent,
    OpConfirmCloseEvent.event_name: OpConfirmCloseEvent,
    AutoStateEvent.event_name: AutoStateEvent,
    AutoProgressEvent.event_name: AutoProgressEvent,
    AutoLenEvent.event_name: AutoLenEvent,
    AutoCoverageEvent.event_name: AutoCoverageEvent,
    AutoRawPointsEvent.event_name: AutoRawPointsEvent,
    AutoRowEvent.event_name: AutoRowEvent,
    AutoStraightnessEvent.event_name: AutoStraightnessEvent,
    AutoPostcalcEvent.event_name: AutoPostcalcEvent,
    AutoClearEvent.event_name: AutoClearEvent,
}


def parse_ui_event(event_name: str, payload: Any) -> KnownUiEvent | None:
    event_cls = UI_EVENT_TYPES.get(str(event_name))
    if event_cls is None:
        return None
    return cast(KnownUiEvent, event_cls.from_payload(payload))


def parse_ui_event_tuple(event: UiEventTuple) -> KnownUiEvent | None:
    event_name, payload = event
    return parse_ui_event(event_name, payload)


__all__ = [
    'AutoClearEvent',
    'AutoCoverageEvent',
    'AutoLenEvent',
    'AutoPostcalcEvent',
    'AutoProgressEvent',
    'AutoRawPointsEvent',
    'AutoRowEvent',
    'AutoStateEvent',
    'AutoStraightnessEvent',
    'GaugeConnEvent',
    'GaugeErrEvent',
    'GaugeOkEvent',
    'GaugeRawEvent',
    'GaugeTxEvent',
    'KnownUiEvent',
    'OpConfirmCloseEvent',
    'OpConfirmShowEvent',
    'PlcErrEvent',
    'PlcGiveupEvent',
    'PlcManualEvent',
    'PlcOkEvent',
    'PlcReadEvent',
    'UI_EVENT_TYPES',
    'UiEventBase',
    'UiEventTuple',
    'parse_ui_event',
    'parse_ui_event_tuple',
]
