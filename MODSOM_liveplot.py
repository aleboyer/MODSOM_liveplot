#!/usr/bin/env python3
"""
modsom_liveplot.py (FAST / STABLE FPS)

Applies the 4 performance fixes (while keeping all existing prints):
1) Fixed-size circular buffers (no unbounded list growth in live mode)
2) PSD updates throttled (default 1 Hz) instead of every GUI refresh
3) Display decimation (limits points sent to pyqtgraph)
4) Avoid storing unbounded raw_payloads in live mode (kept as a small ring)

Behavior:
- File mode: keeps full time series (lists), plots whole file (with decimation if very large)
- Serial/TCP mode: uses circular buffers and plots last window (default 5 s, SB49 10 s)

Inputs:
- --file PATH
- --serial /dev/tty...
- --tcp HOST:PORT

Keeps your header/payload checksum prints exactly in the parser.
"""

import argparse
import threading
import queue
import time
import struct
import signal
import socket
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math

import numpy as np

from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

try:
    import serial  # type: ignore
except ImportError:
    serial = None

# =============================================================================
# Tags
# =============================================================================
VALID_TAGS = {
    b"DCAL",
    b"EFE4",
    b"TTV1",
    b"TTV2",
    b"TTV3",
    b"VNAV",
    b"SB49",
    b"SB41",
    b"ECOP",
    b"SOM3",
}

# =============================================================================
# Time conversion
# =============================================================================
MATLAB_EPOCH_DNUM = 719529.0  # datenum('1970-01-01')
SECONDS_PER_DAY = 86400.0


def posix_to_matlab_dnum(posix_seconds: float) -> float:
    return posix_seconds / SECONDS_PER_DAY + MATLAB_EPOCH_DNUM


# =============================================================================
# ADC conversion helpers (EFE4)
# =============================================================================
ADC_FULL_SCALE_COUNTS = 2 ** 24
ADC_FULL_SCALE_COUNTS_M1 = 2 ** 23
ADC_VREF_TEMP = 2.5
ADC_VREF_SHEAR = 2.5
ADC_VREF_ACCEL = 1.8
ACC_OFFSET = 1.8 / 2
ACC_FACTOR = 0.4

# =============================================================================
# TTV conversion helpers (TTV1, TTV2, TTV3)
# =============================================================================
TTV_ANGLE_VERT2HOR = 53 # the TTV 53˚ to the Horizontal
TTV_ANGLE_B1_VNAVX = 53 # the TTV beam1 to the VNAV X axis
TTV_ANGLE_B2_VNAVX = 53 # the TTV beam2 to the VNAV X axis
TTV_ANGLE_B3_VNAVX = 53 # the TTV beam3 to the VNAV X axis

def bytes3_to_signed_int(b: bytes) -> int:
    if len(b) != 3:
        raise ValueError("bytes3_to_signed_int expects exactly 3 bytes")
    return int.from_bytes(b, byteorder="big", signed=False)


def counts24_to_volts_unipolar(counts: int, full_range: float) -> float:
    return (counts / ADC_FULL_SCALE_COUNTS) * full_range


def counts24_to_volts_bipolar(counts: int, full_range: float) -> float:
    return (counts / ADC_FULL_SCALE_COUNTS_M1) * full_range


def volts_to_g(acc_volts: float) -> float:
    return (acc_volts - ACC_OFFSET) / ACC_FACTOR


# =============================================================================
# TTV constants
# =============================================================================
TTV_SPACE = 0.0382
TTV_ANGLE2VERTICAL = 53
TTV_ANGLE2HORIZONTAL = 90 - TTV_ANGLE2VERTICAL

# =============================================================================
# SBE49 / DCAL
# =============================================================================
C3515_MSCM = 42.914  # mS/cm
C3515_SM = C3515_MSCM / 10.0  # S/m (since 1 S/m = 10 mS/cm)


@dataclass
class SBE49Cal:
    serial_no: str = ""
    temperature_date: str = ""
    conductivity_date: str = ""
    pressure_date: str = ""
    pressure_sn: str = ""
    pressure_range_psia: float = 0.0

    # temperature
    ta0: float = 0.0
    ta1: float = 0.0
    ta2: float = 0.0
    ta3: float = 0.0
    toffset: float = 0.0

    # conductivity
    g: float = 0.0
    h: float = 0.0
    i: float = 0.0
    j: float = 0.0
    tcor: float = 0.0
    pcor: float = 0.0
    cslope: float = 1.0

    # pressure
    pa0: float = 0.0
    pa1: float = 0.0
    pa2: float = 0.0
    ptempa0: float = 0.0
    ptempa1: float = 0.0
    ptempa2: float = 0.0
    ptca0: float = 0.0
    ptca1: float = 0.0
    ptca2: float = 0.0
    ptcb0: float = 0.0
    ptcb1: float = 0.0
    ptcb2: float = 0.0
    poffset: float = 0.0

    valid: bool = False


def parse_dcal_payload(payload: bytes) -> SBE49Cal:
    txt = payload.decode("ascii", errors="ignore")
    cal = SBE49Cal()

    m = re.search(r"SERIAL\s+NO\.\s*([0-9]+)", txt, flags=re.IGNORECASE)
    if m:
        cal.serial_no = m.group(1).strip()

    m = re.search(r"temperature:\s*([0-9A-Za-z\-]+)", txt, flags=re.IGNORECASE)
    if m:
        cal.temperature_date = m.group(1).strip()

    m = re.search(r"conductivity:\s*([0-9A-Za-z\-]+)", txt, flags=re.IGNORECASE)
    if m:
        cal.conductivity_date = m.group(1).strip()

    m = re.search(
        r"pressure\s+S/N\s*=\s*([0-9]+).*?range\s*=\s*([0-9.]+)\s*psia:\s*([0-9A-Za-z\-]+)",
        txt,
        flags=re.IGNORECASE,
    )
    if m:
        cal.pressure_sn = m.group(1).strip()
        try:
            cal.pressure_range_psia = float(m.group(2))
        except Exception:
            cal.pressure_range_psia = 0.0
        cal.pressure_date = m.group(3).strip()

    kv = {}
    for line in txt.splitlines():
        mm = re.search(
            r"^\s*([A-Za-z0-9_]+)\s*=\s*([+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)\s*$", line
        )
        if mm:
            kv[mm.group(1).upper()] = float(mm.group(2))

    cal.ta0 = kv.get("TA0", 0.0)
    cal.ta1 = kv.get("TA1", 0.0)
    cal.ta2 = kv.get("TA2", 0.0)
    cal.ta3 = kv.get("TA3", 0.0)
    cal.toffset = kv.get("TOFFSET", 0.0)

    cal.g = kv.get("G", 0.0)
    cal.h = kv.get("H", 0.0)
    cal.i = kv.get("I", 0.0)
    cal.j = kv.get("J", 0.0)
    cal.pcor = kv.get("CPCOR", 0.0)
    cal.tcor = kv.get("CTCOR", 0.0)
    cal.cslope = kv.get("CSLOPE", 1.0)

    cal.pa0 = kv.get("PA0", 0.0)
    cal.pa1 = kv.get("PA1", 0.0)
    cal.pa2 = kv.get("PA2", 0.0)
    cal.ptca0 = kv.get("PTCA0", 0.0)
    cal.ptca1 = kv.get("PTCA1", 0.0)
    cal.ptca2 = kv.get("PTCA2", 0.0)
    cal.ptcb0 = kv.get("PTCB0", 0.0)
    cal.ptcb1 = kv.get("PTCB1", 0.0)
    cal.ptcb2 = kv.get("PTCB2", 0.0)
    cal.ptempa0 = kv.get("PTEMPA0", 0.0)
    cal.ptempa1 = kv.get("PTEMPA1", 0.0)
    cal.ptempa2 = kv.get("PTEMPA2", 0.0)
    cal.poffset = kv.get("POFFSET", 0.0)

    cal.valid = (
        (cal.ta1 != 0.0)
        and (cal.g != 0.0)
        and (cal.pa1 != 0.0)
        and (cal.ptca0 != 0.0)
        and (cal.ptcb0 != 0.0)
    )
    return cal


def sbe49_raw_to_temperature(T_raw: int, cal: SBE49Cal) -> float:
    mv = (float(T_raw) - 524288.0) / 1.6e7
    denom = (6.144e4 - mv * 5.3e5)
    if denom == 0:
        denom = 1e-12
    r = (mv * 2.295e10 + 9.216e8) / denom
    if r <= 0:
        r = 1e-12
    lr = math.log(r)
    invT = cal.ta0 + cal.ta1 * lr + cal.ta2 * (lr**2) + cal.ta3 * (lr**3)
    if invT == 0:
        invT = 1e-12
    return (1.0 / invT) - 273.15


def sbe49_raw_to_pressure(P_raw: int, PT_raw: int, cal: SBE49Cal) -> float:
    y = float(PT_raw) / 13107.0
    t = cal.ptempa0 + cal.ptempa1 * y + cal.ptempa2 * (y**2)
    x = float(P_raw) - cal.ptca0 - cal.ptca1 * t - cal.ptca2 * (t**2)
    denom = (cal.ptcb0 + cal.ptcb1 * t + cal.ptcb2 * (t**2))
    if denom == 0:
        denom = 1e-12
    n = x * cal.ptcb0 / denom
    P = (cal.pa0 + cal.pa1 * n + cal.pa2 * (n**2) - 14.7) * 0.689476
    return P


def sbe49_raw_to_conductivity(C_raw: int, T_C: float, P_dbar: float, cal: SBE49Cal) -> float:
    f = float(C_raw) / 256.0 / 1000.0
    num = cal.g + cal.h * (f**2) + cal.i * (f**3) + cal.j * (f**4)
    den = 1.0 + cal.tcor * float(T_C) + cal.pcor * float(P_dbar)
    if den == 0:
        den = 1e-12
    return num / den  # S/m


def salinity_from_conductivity_simple(C_Sm: float, T_C: float, P_dbar: float) -> float:
    Rt = float(C_Sm) / C3515_SM
    del_T68 = (T_C * 1.00024) - 15.0
    a0, a1, a2, a3, a4, a5 = 0.0080, -0.1692, 25.3851, 14.0941, -7.0261, 2.7081
    b0, b1, b2, b3, b4, b5 = 0.0005, -0.0056, -0.0066, -0.0375, 0.0636, -0.0144
    k = 0.0162
    if Rt < 0:
        Rt = 0.0
    Rtx = math.sqrt(Rt)
    del_S = (del_T68 / (1.0 + k * del_T68)) * (
        b0 + (b1 + (b2 + (b3 + (b4 + b5 * Rtx) * Rtx) * Rtx) * Rtx) * Rtx
    )
    S = a0 + (a1 + (a2 + (a3 + (a4 + a5 * Rtx) * Rtx) * Rtx) * Rtx) * Rtx
    return S + del_S


# =============================================================================
# Colors
# =============================================================================
EFE4_COLORS = {
    "t1": (255, 0, 0),
    "t2": (255, 165, 0),
    "s1": (0, 0, 255),
    "s2": (0, 255, 255),
    "a1": (0, 255, 0),
    "a2": (255, 0, 255),
    "a3": (139, 69, 19),
}

TTV_TAG_COLORS = {"TTV1": (255, 0, 0), "TTV2": (0, 255, 0), "TTV3": (0, 0, 255)}

VNAV_COLORS = {
    "mag_x": (255, 0, 0),
    "mag_y": (0, 255, 0),
    "mag_z": (0, 0, 255),
    "accel_x": (255, 0, 255),
    "accel_y": (0, 255, 255),
    "accel_z": (255, 255, 0),
    "gyro_x": (255, 165, 0),
    "gyro_y": (128, 0, 128),
    "gyro_z": (0, 128, 0),
}

SB49_COLORS = {"T": (255, 0, 0), "P": (0, 255, 0), "S": (0, 0, 255), "C": (255, 165, 0)}

# =============================================================================
# Fast fixed-size ring buffers
# =============================================================================
class RingBuffer:
    """
    Fixed-size circular buffer for numeric arrays.
    Thread-safe: uses a lock.
    """

    def __init__(self, capacity: int, dtype=np.float64):
        self.capacity = int(capacity)
        self.data = np.empty(self.capacity, dtype=dtype)
        self.n = 0
        self.i = 0
        self.lock = threading.Lock()

    def append(self, x):
        with self.lock:
            self.data[self.i] = x
            self.i = (self.i + 1) % self.capacity
            self.n = min(self.n + 1, self.capacity)

    def extend(self, arr: np.ndarray):
        arr = np.asarray(arr)
        if arr.size == 0:
            return
        with self.lock:
            m = arr.size
            if m >= self.capacity:
                # keep last capacity points
                arr = arr[-self.capacity :]
                m = arr.size
            end = self.i + m
            if end <= self.capacity:
                self.data[self.i : end] = arr
            else:
                k = self.capacity - self.i
                self.data[self.i :] = arr[:k]
                self.data[: end - self.capacity] = arr[k:]
            self.i = (self.i + m) % self.capacity
            self.n = min(self.n + m, self.capacity)

    def get(self) -> np.ndarray:
        with self.lock:
            if self.n == 0:
                return np.array([], dtype=self.data.dtype)
            if self.n < self.capacity:
                return self.data[: self.n].copy()
            # full
            return np.concatenate((self.data[self.i :], self.data[: self.i])).copy()

    def latest(self) -> Optional[float]:
        with self.lock:
            if self.n == 0:
                return None
            idx = (self.i - 1) % self.capacity
            return float(self.data[idx])


class RingBytes:
    """Small ring for bytes payloads (optional, avoids unbounded growth)."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.data: List[Optional[bytes]] = [None] * self.capacity
        self.n = 0
        self.i = 0
        self.lock = threading.Lock()

    def append(self, b: bytes):
        with self.lock:
            self.data[self.i] = b
            self.i = (self.i + 1) % self.capacity
            self.n = min(self.n + 1, self.capacity)


# =============================================================================
# Data structures
# =============================================================================
@dataclass
class Record:
    inst_tag: str
    posix: float
    dnum: float
    payload_size: int
    payload: bytes


class BaseInstrumentData:
    """
    Two modes:
    - file_mode=True  -> store full lists
    - file_mode=False -> store ring buffers (fixed size)
    """

    def __init__(self, file_mode: bool, ring_capacity: int):
        self.file_mode = file_mode
        self.lock = threading.Lock()

        if file_mode:
            self.record_posix: List[float] = []
            self.record_dnum: List[float] = []
            self.sample_posix: List[float] = []
            self.sample_dnum: List[float] = []
        else:
            self.record_posix = RingBuffer(max(128, ring_capacity // 10), dtype=np.float64)
            self.record_dnum = RingBuffer(max(128, ring_capacity // 10), dtype=np.float64)
            self.sample_posix = RingBuffer(ring_capacity, dtype=np.float64)
            self.sample_dnum = RingBuffer(ring_capacity, dtype=np.float64)

    def _append_record_time(self, posix: float, dnum: float):
        if self.file_mode:
            self.record_posix.append(posix)
            self.record_dnum.append(dnum)
        else:
            self.record_posix.append(posix)
            self.record_dnum.append(dnum)

    def _append_sample_time(self, posix: float, dnum: float):
        if self.file_mode:
            self.sample_posix.append(posix)
            self.sample_dnum.append(dnum)
        else:
            self.sample_posix.append(posix)
            self.sample_dnum.append(dnum)

    def get_sample_posix(self) -> np.ndarray:
        return np.asarray(self.sample_posix, dtype=float) if self.file_mode else self.sample_posix.get()


class DCALData(BaseInstrumentData):
    def __init__(self, file_mode: bool, ring_capacity: int):
        super().__init__(file_mode, ring_capacity)
        self.raw_text: List[str] = []
        self.cal: Optional[SBE49Cal] = None


class EFE4Data(BaseInstrumentData):
    def __init__(self, file_mode: bool, ring_capacity: int):
        super().__init__(file_mode, ring_capacity)
        if file_mode:
            self.t1: List[float] = []
            self.t2: List[float] = []
            self.s1: List[float] = []
            self.s2: List[float] = []
            self.a1: List[float] = []
            self.a2: List[float] = []
            self.a3: List[float] = []
            self.raw_payloads: List[bytes] = []
        else:
            self.t1 = RingBuffer(ring_capacity, dtype=np.float64)
            self.t2 = RingBuffer(ring_capacity, dtype=np.float64)
            self.s1 = RingBuffer(ring_capacity, dtype=np.float64)
            self.s2 = RingBuffer(ring_capacity, dtype=np.float64)
            self.a1 = RingBuffer(ring_capacity, dtype=np.float64)
            self.a2 = RingBuffer(ring_capacity, dtype=np.float64)
            self.a3 = RingBuffer(ring_capacity, dtype=np.float64)
            self.raw_payloads = RingBytes(50)  # keep last 50 payloads only


class TTVData(BaseInstrumentData):
    def __init__(self, file_mode: bool, ring_capacity: int):
        super().__init__(file_mode, ring_capacity)
        if file_mode:
            self.tof_up: List[float] = []
            self.tof_down: List[float] = []
            self.dtof: List[float] = []
            self.errorcode: List[int] = []
            self.upstream_adcpeak: List[int] = []
            self.downstream_adcpeak: List[int] = []
            self.raw_payloads: List[bytes] = []
        else:
            self.tof_up = RingBuffer(ring_capacity, dtype=np.float64)
            self.tof_down = RingBuffer(ring_capacity, dtype=np.float64)
            self.dtof = RingBuffer(ring_capacity, dtype=np.float64)
            self.errorcode = RingBuffer(ring_capacity, dtype=np.float64)  # store as float for plotting
            self.upstream_adcpeak = RingBuffer(ring_capacity, dtype=np.float64)
            self.downstream_adcpeak = RingBuffer(ring_capacity, dtype=np.float64)
            self.raw_payloads = RingBytes(50)

class TTVProcessedData(BaseInstrumentData):
    def __init__(self, file_mode: bool, ring_capacity: int):
        super().__init__(file_mode, ring_capacity)
        if file_mode:
            self.ttv1_posix: List[float] = []
            self.ttv2_posix: List[float] = []
            self.ttv3_posix: List[float] = []
            self.beam1_vel: List[float] = []
            self.beam2_vel: List[float] = []
            self.beam3_vel: List[float] = []
            self.velZ1: List[float] = []
            self.velZ2: List[float] = []
            self.velZ3: List[float] = []
            self.velU1: List[float] = []
            self.velU2: List[float] = []
            self.velU3: List[float] = []
        else:
            self.ttv1_posix = RingBuffer(ring_capacity, dtype=np.float64)
            self.ttv2_posix = RingBuffer(ring_capacity, dtype=np.float64)
            self.ttv3_posix = RingBuffer(ring_capacity, dtype=np.float64)
            self.beam1_vel  = RingBuffer(ring_capacity, dtype=np.float64)
            self.beam2_vel  = RingBuffer(ring_capacity, dtype=np.float64)
            self.beam3_vel  = RingBuffer(ring_capacity, dtype=np.float64)
            self.velZ1      = RingBuffer(ring_capacity, dtype=np.float64)
            self.velZ2      = RingBuffer(ring_capacity, dtype=np.float64)
            self.velZ3      = RingBuffer(ring_capacity, dtype=np.float64)
            self.velU1      = RingBuffer(ring_capacity, dtype=np.float64)
            self.velU2      = RingBuffer(ring_capacity, dtype=np.float64)
            self.velU3      = RingBuffer(ring_capacity, dtype=np.float64)


class SB49Data(BaseInstrumentData):
    def __init__(self, file_mode: bool, ring_capacity: int):
        super().__init__(file_mode, ring_capacity)
        if file_mode:
            self.T: List[float] = []
            self.P: List[float] = []
            self.C: List[float] = []
            self.S: List[float] = []
            self.raw_payloads: List[bytes] = []
        else:
            self.T = RingBuffer(ring_capacity, dtype=np.float64)
            self.P = RingBuffer(ring_capacity, dtype=np.float64)
            self.C = RingBuffer(ring_capacity, dtype=np.float64)
            self.S = RingBuffer(ring_capacity, dtype=np.float64)
            self.raw_payloads = RingBytes(50)


class SB41Data(BaseInstrumentData):
    def __init__(self, file_mode: bool, ring_capacity: int):
        super().__init__(file_mode, ring_capacity)
        self.raw_payloads = [] if file_mode else RingBytes(50)


class VNAVData(BaseInstrumentData):
    def __init__(self, file_mode: bool, ring_capacity: int):
        super().__init__(file_mode, ring_capacity)
        if file_mode:
            self.mag_x: List[float] = []
            self.mag_y: List[float] = []
            self.mag_z: List[float] = []
            self.accel_x: List[float] = []
            self.accel_y: List[float] = []
            self.accel_z: List[float] = []
            self.gyro_x: List[float] = []
            self.gyro_y: List[float] = []
            self.gyro_z: List[float] = []
            self.raw_payloads: List[bytes] = []
        else:
            self.mag_x = RingBuffer(ring_capacity, dtype=np.float64)
            self.mag_y = RingBuffer(ring_capacity, dtype=np.float64)
            self.mag_z = RingBuffer(ring_capacity, dtype=np.float64)
            self.accel_x = RingBuffer(ring_capacity, dtype=np.float64)
            self.accel_y = RingBuffer(ring_capacity, dtype=np.float64)
            self.accel_z = RingBuffer(ring_capacity, dtype=np.float64)
            self.gyro_x = RingBuffer(ring_capacity, dtype=np.float64)
            self.gyro_y = RingBuffer(ring_capacity, dtype=np.float64)
            self.gyro_z = RingBuffer(ring_capacity, dtype=np.float64)
            self.raw_payloads = RingBytes(50)


class ECOPData(BaseInstrumentData):
    def __init__(self, file_mode: bool, ring_capacity: int):
        super().__init__(file_mode, ring_capacity)
        self.raw_payloads = [] if file_mode else RingBytes(50)


class SOM3Data(BaseInstrumentData):
    def __init__(self, file_mode: bool, ring_capacity: int):
        super().__init__(file_mode, ring_capacity)
        self.raw_payloads = [] if file_mode else RingBytes(50)


# =============================================================================
# Reader thread: file / serial / tcp socket
# =============================================================================
class ByteSourceThread(threading.Thread):
    def __init__(
        self,
        source,
        mode: str,  # "file" | "serial" | "tcp"
        out_queue: queue.Queue,
        stop_event: threading.Event,
        chunk_size: int = 4096,
        dcal_command: Optional[bytes] = b"sbe.dcal\r\n",
        dcal_command_delay_s: float = 0.05,
        start_command: Optional[bytes] = b"som.start\r\n",
        stop_command: Optional[bytes] = b"som.stop\r\n",
        start_command_delay_s: float = 0.05,
        clear_input_before_start: bool = True,
    ):
        super().__init__(daemon=True)
        self.source = source
        self.mode = mode
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.chunk_size = chunk_size
        self.dcal_command = dcal_command
        self.start_command = start_command
        self.dcal_command_delay_s = dcal_command_delay_s
        self.stop_command = stop_command
        self.start_command_delay_s = start_command_delay_s
        self.clear_input_before_start = clear_input_before_start

    def _send_serial_cmd(self, cmd: Optional[bytes]):
        if self.mode != "serial" or not cmd:
            return
        try:
            self.source.write(cmd)
            if hasattr(self.source, "flush"):
                self.source.flush()
        except Exception as e:
            print(f"[ByteSourceThread] Warning: failed to send serial command {cmd!r}: {e}")

    def run(self):
        try:
            if self.mode == "serial" and self.start_command:
                try:
                    if self.clear_input_before_start and hasattr(self.source, "reset_input_buffer"):
                        self.source.reset_input_buffer()
                    self._send_serial_cmd(self.dcal_command)
                    if self.dcal_command_delay_s > 0:
                        time.sleep(self.dcal_command_delay_s)
                    self._send_serial_cmd(self.start_command)
                    print(f"[som.start] send command to serial")
                    if self.start_command_delay_s > 0:
                        time.sleep(self.start_command_delay_s)
                except Exception as e:
                    print(f"[ByteSourceThread] Warning: failed to send start command: {e}")

            while not self.stop_event.is_set():
                if self.mode == "tcp":
                    try:
                        data = self.source.recv(self.chunk_size)
                    except socket.timeout:
                        data = b""
                else:
                    data = self.source.read(self.chunk_size)

                if not data:
                    if self.mode == "file":
                        break
                    time.sleep(0.01)
                    continue

                self.out_queue.put(data)

        except Exception as e:
            print(f"[ByteSourceThread] Error: {e}")
        finally:
            if self.mode == "serial" and self.stop_command:
                self._send_serial_cmd(self.stop_command)
            self.out_queue.put(None)


# =============================================================================
# Parser: state machine (prints kept)
# =============================================================================
class EpsiStateMachineParser:
    """
    $TAG tttttttttttttttt AAAAAAAA *CC <payload> *PP
    """

    STATE_SYNC = 0
    STATE_TAG = 1
    STATE_HEADER = 2
    STATE_PAYLOAD = 3

    HEADER_LEN = 1 + 4 + 16 + 8 + 1 + 2  # "$" + TAG + ts16 + size8 + "*" + cc2

    def __init__(self, record_queue: queue.Queue):
        self.buffer = bytearray()
        self.state = self.STATE_SYNC
        self.current_tag: Optional[bytes] = None
        self.current_timestamp_ms: Optional[int] = None
        self.current_payload_size: Optional[int] = None
        self.record_queue = record_queue

    def feed(self, data: bytes):
        self.buffer.extend(data)
        while True:
            if self.state == self.STATE_SYNC:
                if not self._phase_sync():
                    break
            elif self.state == self.STATE_TAG:
                if not self._phase_tag():
                    break
            elif self.state == self.STATE_HEADER:
                if not self._phase_header():
                    break
            elif self.state == self.STATE_PAYLOAD:
                if not self._phase_payload():
                    break
            else:
                self.state = self.STATE_SYNC

    def _phase_sync(self) -> bool:
        idx = self.buffer.find(b"$")
        if idx == -1:
            self.buffer.clear()
            return False
        if idx > 0:
            del self.buffer[:idx]
        if len(self.buffer) < 1 + 4:
            return False
        self.state = self.STATE_TAG
        return True

    def _phase_tag(self) -> bool:
        if len(self.buffer) < 1 + 4:
            return False

        if self.buffer[0] != ord("$"):
            del self.buffer[0]
            self.state = self.STATE_SYNC
            return True

        tag_bytes = bytes(self.buffer[1:5])
        if tag_bytes not in VALID_TAGS:
            del self.buffer[0]
            self.state = self.STATE_SYNC
            return True

        self.current_tag = tag_bytes
        self.state = self.STATE_HEADER
        return True

    def _phase_header(self) -> bool:
        if len(self.buffer) < self.HEADER_LEN:
            return False

        header = self.buffer[: self.HEADER_LEN]

        if header[0] != ord("$"):
            del self.buffer[0]
            self.state = self.STATE_SYNC
            return True

        if header[29] != ord("*"):
            print(f"[HEADER] malformed header (no '*'): {header!r}")
            del self.buffer[0]
            self.state = self.STATE_SYNC
            return True

        ts_hex = header[5:21].decode("ascii", errors="ignore")
        size_hex = header[21:29].decode("ascii", errors="ignore")
        cksum_hex = header[30:32].decode("ascii", errors="ignore")

        try:
            timestamp_ms = int(ts_hex, 16)
            payload_size = int(size_hex, 16)
            published_cksum = int(cksum_hex, 16)
        except ValueError:
            print(f"[HEADER] invalid hex fields in header: {header!r}")
            del self.buffer[0]
            self.state = self.STATE_SYNC
            return True

        computed_cksum = 0
        for b in header[0:29]:
            computed_cksum ^= b
        computed_cksum &= 0xFF

        tag_str = self.current_tag.decode("ascii") if self.current_tag else "????"
        if computed_cksum == published_cksum:
            self.current_timestamp_ms = timestamp_ms
            self.current_payload_size = payload_size
            del self.buffer[: self.HEADER_LEN]
            self.state = self.STATE_PAYLOAD
            print(f"[HEADER]  tag={tag_str} GOOD checksum")
        else:
            print(
                f"[HEADER] tag={tag_str} BAD checksum "
                f"(computed=0x{computed_cksum:02X}, published=0x{published_cksum:02X})"
            )
            del self.buffer[0]
            self.state = self.STATE_SYNC

        return True

    def _phase_payload(self) -> bool:
        if self.current_payload_size is None or self.current_tag is None or self.current_timestamp_ms is None:
            self.state = self.STATE_SYNC
            return True

        needed = self.current_payload_size + 1 + 2  # payload + "*" + 2 hex
        if len(self.buffer) < needed:
            return False

        payload = self.buffer[: self.current_payload_size]
        star_byte = self.buffer[self.current_payload_size]
        cksum_bytes = self.buffer[self.current_payload_size + 1 : self.current_payload_size + 3]

        tag_str = self.current_tag.decode("ascii")

        if star_byte != ord("*"):
            print(f"[PAYLOAD] tag={tag_str} malformed payload (no '*')")
            del self.buffer[0]
            self._reset_current()
            self.state = self.STATE_SYNC
            return True

        try:
            published_cksum = int(cksum_bytes.decode("ascii", errors="ignore"), 16)
        except ValueError:
            print(f"[PAYLOAD] tag={tag_str} invalid payload checksum hex: {cksum_bytes!r}")
            del self.buffer[0]
            self._reset_current()
            self.state = self.STATE_SYNC
            return True

        computed_cksum = 0
        for b in payload:
            computed_cksum ^= b
        computed_cksum &= 0xFF

        if computed_cksum == published_cksum:
            posix_sec = self.current_timestamp_ms / 1000.0
            dnum = posix_to_matlab_dnum(posix_sec)
            rec = Record(
                inst_tag=tag_str,
                posix=posix_sec,
                dnum=dnum,
                payload_size=self.current_payload_size,
                payload=bytes(payload),
            )
            self.record_queue.put(rec)
            print(f"[PAYLOAD] tag={tag_str} GOOD checksum")
        else:
            print(f"[PAYLOAD] tag={tag_str} BAD payload checksum")

        del self.buffer[:needed]
        self._reset_current()
        self.state = self.STATE_SYNC
        return True

    def _reset_current(self):
        self.current_tag = None
        self.current_timestamp_ms = None
        self.current_payload_size = None


# =============================================================================
# Parser thread
# =============================================================================
class ParserThread(threading.Thread):
    def __init__(
        self, byte_queue: queue.Queue, parser: EpsiStateMachineParser, record_queue: queue.Queue, stop_event: threading.Event
    ):
        super().__init__(daemon=True)
        self.byte_queue = byte_queue
        self.parser = parser
        self.record_queue = record_queue
        self.stop_event = stop_event

    def run(self):
        while not self.stop_event.is_set():
            chunk = self.byte_queue.get()
            if chunk is None:
                break
            self.parser.feed(chunk)
        self.record_queue.put(None)


# =============================================================================
# Record processing
# =============================================================================
class RecordProcessorThread(threading.Thread):
    """
    Uses:
    - file_mode=True  => lists (keeps everything)
    - file_mode=False => ring buffers (keeps last N only)
    """

    def __init__(self, in_queue: queue.Queue, stop_event: threading.Event, file_mode: bool):
        super().__init__(daemon=True)
        self.in_queue = in_queue
        self.stop_event = stop_event
        self.file_mode = file_mode

        # Ring capacities (live mode): choose safely large; only last window plotted anyway.
        # You can tune these if you want.
        cap_efe4 = 30000   # ~60s @ 500 Hz
        cap_ttv = 10000    # ~100s @ 100 Hz
        cap_vnav = 10000
        cap_sb49 = 5000
        cap_default = 5000

        self.instruments: Dict[str, BaseInstrumentData] = {
            "DCAL": DCALData(file_mode, 256),
            "EFE4": EFE4Data(file_mode, cap_efe4 if not file_mode else cap_default),
            "TTV1": TTVData(file_mode, cap_ttv),
            "TTV2": TTVData(file_mode, cap_ttv),
            "TTV3": TTVData(file_mode, cap_ttv),
            "SB49": SB49Data(file_mode, cap_sb49),
            "SB41": SB41Data(file_mode, cap_default),
            "VNAV": VNAVData(file_mode, cap_vnav),
            "ECOP": ECOPData(file_mode, cap_default),
            "SOM3": SOM3Data(file_mode, cap_default),
        }

        self.sb49_cal: Optional[SBE49Cal] = None

        self.process_data: Dict[str,BaseInstrumentData] = {"TTV": TTVProcessedData(file_mode, cap_ttv)}

    def run(self):
        while not self.stop_event.is_set():
            rec = self.in_queue.get()
            if rec is None:
                break
            self._process_record(rec)

    def _process_record(self, rec: Record):
        tag = rec.inst_tag
        if tag not in self.instruments:
            return

        if tag == "DCAL":
            self._parse_dcal_record(rec, self.instruments["DCAL"])  # type: ignore[arg-type]
        elif tag == "EFE4":
            self._parse_efe4_record(rec, self.instruments["EFE4"])  # type: ignore[arg-type]
        elif tag in ("TTV1", "TTV2", "TTV3"):
            self._parse_ttv_record(rec, self.instruments[tag], tag)  # type: ignore[arg-type]
        elif tag == "SB49":
            self._parse_sb49_record(rec, self.instruments["SB49"])  # type: ignore[arg-type]
        elif tag == "SB41":
            self._parse_sb41_record(rec, self.instruments["SB41"])  # type: ignore[arg-type]
        elif tag == "VNAV":
            self._parse_vnav_record(rec, self.instruments["VNAV"])  # type: ignore[arg-type]
        elif tag == "ECOP":
            self._parse_ecop_record(rec, self.instruments["ECOP"])  # type: ignore[arg-type]
        elif tag == "SOM3":
            self._parse_som3_record(rec, self.instruments["SOM3"])  # type: ignore[arg-type]

    def _parse_dcal_record(self, rec: Record, inst: DCALData):
        inst._append_record_time(rec.posix, rec.dnum)
        txt = rec.payload.decode("ascii", errors="ignore")
        inst.raw_text.append(txt)
        cal = parse_dcal_payload(rec.payload)
        inst.cal = cal
        self.sb49_cal = cal
        if cal.valid:
            print(f"[DCAL] Parsed SBE49 cal: SN={cal.serial_no} (valid)")
        else:
            print(f"[DCAL] Parsed SBE49 cal: SN={cal.serial_no} (NOT valid?)")

    def _parse_efe4_record(self, rec: Record, inst: EFE4Data):
        inst._append_record_time(rec.posix, rec.dnum)
        if inst.file_mode:
            inst.raw_payloads.append(rec.payload)
        else:
            inst.raw_payloads.append(rec.payload)

        payload = rec.payload
        SAMPLE_BYTES = 8 + 3 * 7
        n_samples = len(payload) // SAMPLE_BYTES
        if n_samples == 0:
            if len(payload) > 0:
                print(f"[EFE4] Warning: payload length {len(payload)} not a multiple of {SAMPLE_BYTES}")
            return

        # Batch parse into numpy for faster ring extension (live mode)
        if not inst.file_mode:
            # timestamps
            ts = np.empty(n_samples, dtype=np.float64)
            t1 = np.empty(n_samples, dtype=np.float64)
            t2 = np.empty(n_samples, dtype=np.float64)
            s1 = np.empty(n_samples, dtype=np.float64)
            s2 = np.empty(n_samples, dtype=np.float64)
            a1 = np.empty(n_samples, dtype=np.float64)
            a2 = np.empty(n_samples, dtype=np.float64)
            a3 = np.empty(n_samples, dtype=np.float64)

            for i in range(n_samples):
                off = i * SAMPLE_BYTES
                sample_ts_bytes = payload[off : off + 8]
                chan_bytes = payload[off + 8 : off + SAMPLE_BYTES]
                if len(sample_ts_bytes) != 8 or len(chan_bytes) != 21:
                    ts = ts[:i]
                    t1 = t1[:i]
                    t2 = t2[:i]
                    s1 = s1[:i]
                    s2 = s2[:i]
                    a1 = a1[:i]
                    a2 = a2[:i]
                    a3 = a3[:i]
                    break

                sample_ts_ms = int.from_bytes(sample_ts_bytes, byteorder="little", signed=False)
                sample_posix = sample_ts_ms / 1000.0
                ts[i] = sample_posix

                c_t1 = bytes3_to_signed_int(chan_bytes[0:3])
                c_t2 = bytes3_to_signed_int(chan_bytes[3:6])
                c_s1 = bytes3_to_signed_int(chan_bytes[6:9])
                c_s2 = bytes3_to_signed_int(chan_bytes[9:12])
                c_a1 = bytes3_to_signed_int(chan_bytes[12:15])
                c_a2 = bytes3_to_signed_int(chan_bytes[15:18])
                c_a3 = bytes3_to_signed_int(chan_bytes[18:21])

                t1[i] = counts24_to_volts_unipolar(c_t1, ADC_VREF_TEMP)
                t2[i] = counts24_to_volts_unipolar(c_t2, ADC_VREF_TEMP)
                s1[i] = counts24_to_volts_bipolar(c_s1, ADC_VREF_SHEAR)
                s2[i] = counts24_to_volts_bipolar(c_s2, ADC_VREF_SHEAR)
                a1[i] = volts_to_g(counts24_to_volts_unipolar(c_a1, ADC_VREF_ACCEL))
                a2[i] = volts_to_g(counts24_to_volts_unipolar(c_a2, ADC_VREF_ACCEL))
                a3[i] = volts_to_g(counts24_to_volts_unipolar(c_a3, ADC_VREF_ACCEL))

            # extend rings
            inst.sample_posix.extend(ts)
            inst.sample_dnum.extend(ts / SECONDS_PER_DAY + MATLAB_EPOCH_DNUM)
            inst.t1.extend(t1)
            inst.t2.extend(t2)
            inst.s1.extend(s1)
            inst.s2.extend(s2)
            inst.a1.extend(a1)
            inst.a2.extend(a2)
            inst.a3.extend(a3)
            return

        # file_mode: original list appends
        for i in range(n_samples):
            offset = i * SAMPLE_BYTES
            sample_ts_bytes = payload[offset : offset + 8]
            chan_bytes = payload[offset + 8 : offset + SAMPLE_BYTES]
            if len(sample_ts_bytes) != 8 or len(chan_bytes) != 21:
                break

            sample_ts_ms = int.from_bytes(sample_ts_bytes, byteorder="little", signed=False)
            sample_posix = sample_ts_ms / 1000.0
            inst.sample_posix.append(sample_posix)
            inst.sample_dnum.append(posix_to_matlab_dnum(sample_posix))

            c_t1 = bytes3_to_signed_int(chan_bytes[0:3])
            c_t2 = bytes3_to_signed_int(chan_bytes[3:6])
            c_s1 = bytes3_to_signed_int(chan_bytes[6:9])
            c_s2 = bytes3_to_signed_int(chan_bytes[9:12])
            c_a1 = bytes3_to_signed_int(chan_bytes[12:15])
            c_a2 = bytes3_to_signed_int(chan_bytes[15:18])
            c_a3 = bytes3_to_signed_int(chan_bytes[18:21])

            inst.t1.append(counts24_to_volts_unipolar(c_t1, ADC_VREF_TEMP))
            inst.t2.append(counts24_to_volts_unipolar(c_t2, ADC_VREF_TEMP))
            inst.s1.append(counts24_to_volts_bipolar(c_s1, ADC_VREF_SHEAR))
            inst.s2.append(counts24_to_volts_bipolar(c_s2, ADC_VREF_SHEAR))
            inst.a1.append(volts_to_g(counts24_to_volts_unipolar(c_a1, ADC_VREF_ACCEL)))
            inst.a2.append(volts_to_g(counts24_to_volts_unipolar(c_a2, ADC_VREF_ACCEL)))
            inst.a3.append(volts_to_g(counts24_to_volts_unipolar(c_a3, ADC_VREF_ACCEL)))

    def _parse_ttv_record(self, rec: Record, inst: TTVData, tag: str):
        inst._append_record_time(rec.posix, rec.dnum)
        if inst.file_mode:
            inst.raw_payloads.append(rec.payload)
        else:
            inst.raw_payloads.append(rec.payload)

        payload = rec.payload
        PACKET_SIZE = 16 + 4 + 4 + 4 + 1 + 2 + 2 + 2  # 35 bytes
        if len(payload) < PACKET_SIZE:
            return

        n_packets = len(payload) // PACKET_SIZE

        if not inst.file_mode:
            ts = np.empty(n_packets, dtype=np.float64)
            tof_up = np.empty(n_packets, dtype=np.float64)
            tof_dn = np.empty(n_packets, dtype=np.float64)
            dtof = np.empty(n_packets, dtype=np.float64)
            err = np.empty(n_packets, dtype=np.float64)
            upk = np.empty(n_packets, dtype=np.float64)
            dpk = np.empty(n_packets, dtype=np.float64)

            for i in range(n_packets):
                offset = i * PACKET_SIZE
                chunk = payload[offset : offset + PACKET_SIZE]
                if len(chunk) < PACKET_SIZE:
                    ts = ts[:i]
                    tof_up = tof_up[:i]
                    tof_dn = tof_dn[:i]
                    dtof = dtof[:i]
                    err = err[:i]
                    upk = upk[:i]
                    dpk = dpk[:i]
                    break

                ts_hex_bytes = chunk[0:16]
                try:
                    ts_ms = int(ts_hex_bytes.decode("ascii", errors="ignore"), 16)
                except ValueError:
                    continue

                sp = ts_ms / 1000.0
                ts[i] = sp

                idx = 16
                u = struct.unpack(">f", chunk[idx : idx + 4])[0]
                idx += 4
                d = struct.unpack(">f", chunk[idx : idx + 4])[0]
                idx += 4
                dt = struct.unpack(">f", chunk[idx : idx + 4])[0]
                idx += 4
                e = chunk[idx]
                idx += 1
                up = struct.unpack(">H", chunk[idx : idx + 2])[0]
                idx += 2
                dn = struct.unpack(">H", chunk[idx : idx + 2])[0]
                idx += 2

                tof_up[i] = float(u)
                tof_dn[i] = float(d)
                dtof[i] = float(dt)
                err[i] = float(int(e))
                upk[i] = float(int(up))
                dpk[i] = float(int(dn))

            inst.sample_posix.extend(ts)
            inst.sample_dnum.extend(ts / SECONDS_PER_DAY + MATLAB_EPOCH_DNUM)
            inst.tof_up.extend(tof_up)
            inst.tof_down.extend(tof_dn)
            inst.dtof.extend(dtof)
            inst.errorcode.extend(err)
            inst.upstream_adcpeak.extend(upk)
            inst.downstream_adcpeak.extend(dpk)
            return

        # file_mode lists
        for i in range(n_packets):
            offset = i * PACKET_SIZE
            chunk = payload[offset : offset + PACKET_SIZE]
            if len(chunk) < PACKET_SIZE:
                break
            ts_hex_bytes = chunk[0:16]
            try:
                ts_ms = int(ts_hex_bytes.decode("ascii", errors="ignore"), 16)
            except ValueError:
                continue
            sample_posix = ts_ms / 1000.0
            inst.sample_posix.append(sample_posix)
            inst.sample_dnum.append(posix_to_matlab_dnum(sample_posix))

            idx = 16
            tof_up = struct.unpack(">f", chunk[idx : idx + 4])[0]
            idx += 4
            tof_down = struct.unpack(">f", chunk[idx : idx + 4])[0]
            idx += 4
            dtof = struct.unpack(">f", chunk[idx : idx + 4])[0]
            idx += 4
            errorcode = chunk[idx]
            idx += 1
            upstream_adcpeak = struct.unpack(">H", chunk[idx : idx + 2])[0]
            idx += 2
            downstream_adcpeak = struct.unpack(">H", chunk[idx : idx + 2])[0]
            idx += 2

            # if tof_up <= 0 or tof_down <= 0:
            #     beam_vel = 0.0
            # else:
            #     beam_vel = (TTV_SPACE / 2.0) * dtof / (tof_up * tof_down)

            inst.tof_up.append(float(tof_up))
            inst.tof_down.append(float(tof_down))
            inst.dtof.append(float(dtof))
            inst.errorcode.append(int(errorcode))
            inst.upstream_adcpeak.append(int(upstream_adcpeak))
            inst.downstream_adcpeak.append(int(downstream_adcpeak))


    def _parse_sb49_record(self, rec: Record, inst: SB49Data):
        inst._append_record_time(rec.posix, rec.dnum)
        if inst.file_mode:
            inst.raw_payloads.append(rec.payload)
        else:
            inst.raw_payloads.append(rec.payload)

        cal = self.sb49_cal
        if cal is None or not cal.valid:
            print("[SB49] Warning: no valid DCAL parsed yet; storing raw only.")
            return

        payload = rec.payload
        ELEMENT_TS_LEN = 16
        RAW_LEN = 24
        ELEMENT_LEN = ELEMENT_TS_LEN + RAW_LEN

        if len(payload) < ELEMENT_LEN:
            return

        n_el = len(payload) // ELEMENT_LEN
        if len(payload) % ELEMENT_LEN != 0:
            print(f"[SB49] Warning: payload not divisible by {ELEMENT_LEN}; using first {n_el} elements.")

        # live mode: batch
        if not inst.file_mode:
            ts = np.empty(n_el, dtype=np.float64)
            TT = np.empty(n_el, dtype=np.float64)
            PP = np.empty(n_el, dtype=np.float64)
            CC = np.empty(n_el, dtype=np.float64)
            SS = np.empty(n_el, dtype=np.float64)

            j = 0
            for k in range(n_el):
                off = k * ELEMENT_LEN
                ts_hex = payload[off : off + 16]
                raw = payload[off + 16 : off + 16 + RAW_LEN]
                try:
                    ts_ms = int(ts_hex.decode("ascii", errors="ignore"), 16)
                except ValueError:
                    continue

                raw_str = raw.decode("ascii", errors="ignore")
                core = raw_str[:22]
                if len(core) < 22:
                    continue
                try:
                    T_raw = int(core[0:6], 16)
                    C_raw = int(core[6:12], 16)
                    P_raw = int(core[12:18], 16)
                    PT_raw = int(core[18:22], 16)
                except ValueError:
                    continue

                sp = ts_ms / 1000.0
                T_C = sbe49_raw_to_temperature(T_raw, cal)
                P_dbar = sbe49_raw_to_pressure(P_raw, PT_raw, cal)
                C_Sm = sbe49_raw_to_conductivity(C_raw, T_C, P_dbar, cal)
                S_psu = salinity_from_conductivity_simple(C_Sm, T_C, P_dbar)

                ts[j] = sp
                TT[j] = T_C
                PP[j] = P_dbar
                CC[j] = C_Sm
                SS[j] = S_psu
                j += 1

            ts = ts[:j]
            TT = TT[:j]
            PP = PP[:j]
            CC = CC[:j]
            SS = SS[:j]

            inst.sample_posix.extend(ts)
            inst.sample_dnum.extend(ts / SECONDS_PER_DAY + MATLAB_EPOCH_DNUM)
            inst.T.extend(TT)
            inst.P.extend(PP)
            inst.C.extend(CC)
            inst.S.extend(SS)
            return

        # file mode lists
        for k in range(n_el):
            off = k * ELEMENT_LEN
            ts_hex = payload[off : off + 16]
            raw = payload[off + 16 : off + 16 + RAW_LEN]
            try:
                ts_ms = int(ts_hex.decode("ascii", errors="ignore"), 16)
            except ValueError:
                continue

            raw_str = raw.decode("ascii", errors="ignore")
            core = raw_str[:22]
            if len(core) < 22:
                continue
            try:
                T_raw = int(core[0:6], 16)
                C_raw = int(core[6:12], 16)
                P_raw = int(core[12:18], 16)
                PT_raw = int(core[18:22], 16)
            except ValueError:
                continue

            sample_posix = ts_ms / 1000.0
            inst.sample_posix.append(sample_posix)
            inst.sample_dnum.append(posix_to_matlab_dnum(sample_posix))

            T_C = sbe49_raw_to_temperature(T_raw, cal)
            P_dbar = sbe49_raw_to_pressure(P_raw, PT_raw, cal)
            C_Sm = sbe49_raw_to_conductivity(C_raw, T_C, P_dbar, cal)
            S_psu = salinity_from_conductivity_simple(C_Sm, T_C, P_dbar)

            inst.T.append(float(T_C))
            inst.P.append(float(P_dbar))
            inst.C.append(float(C_Sm))
            inst.S.append(float(S_psu))

    def _parse_sb41_record(self, rec: Record, inst: SB41Data):
        inst._append_record_time(rec.posix, rec.dnum)
        if inst.file_mode:
            inst.raw_payloads.append(rec.payload)  # type: ignore[attr-defined]
        else:
            inst.raw_payloads.append(rec.payload)  # type: ignore[attr-defined]

    def _parse_vnav_record(self, rec: Record, inst: VNAVData):
        inst._append_record_time(rec.posix, rec.dnum)
        if inst.file_mode:
            inst.raw_payloads.append(rec.payload)
        else:
            inst.raw_payloads.append(rec.payload)

        payload = rec.payload
        n = len(payload)
        i = 0

        # live mode: we still parse sequentially (strings), but only store in rings
        while i + 16 + 1 <= n:
            ts_bytes = payload[i : i + 16]
            try:
                ts_ms = int(ts_bytes.decode("ascii", errors="ignore"), 16)
            except ValueError:
                i += 1
                continue

            tag_pos = i + 16
            if tag_pos >= n or payload[tag_pos] != ord("$"):
                i += 1
                continue

            star_idx = payload.find(b"*", tag_pos)
            if star_idx == -1 or star_idx + 2 > n:
                break

            body_for_cksum = payload[tag_pos + 1 : star_idx]
            computed_cksum = 0
            for b in body_for_cksum:
                computed_cksum ^= b
            computed_cksum &= 0xFF

            cksum_bytes = payload[star_idx + 1 : star_idx + 3]
            try:
                published_cksum = int(cksum_bytes.decode("ascii", errors="ignore"), 16)
            except ValueError:
                i = star_idx + 3
                continue

            # keep behavior: do not print here; original code just "pass"
            _ = (computed_cksum != published_cksum)

            msg_bytes = payload[tag_pos:star_idx]
            msg_str = msg_bytes.decode("ascii", errors="ignore")
            fields = msg_str.split(",")
            if len(fields) < 10:
                i = star_idx + 3
                continue

            try:
                magx = float(fields[1])
                magy = float(fields[2])
                magz = float(fields[3])
                accelx = float(fields[4])
                accely = float(fields[5])
                accelz = float(fields[6])
                gyrox = float(fields[7])
                gyroy = float(fields[8])
                gyroz = float(fields[9])
            except ValueError:
                i = star_idx + 3
                continue

            sample_posix = ts_ms / 1000.0
            sample_dnum = posix_to_matlab_dnum(sample_posix)

            if inst.file_mode:
                inst.sample_posix.append(sample_posix)
                inst.sample_dnum.append(sample_dnum)
                inst.mag_x.append(magx)
                inst.mag_y.append(magy)
                inst.mag_z.append(magz)
                inst.accel_x.append(accelx)
                inst.accel_y.append(accely)
                inst.accel_z.append(accelz)
                inst.gyro_x.append(gyrox)
                inst.gyro_y.append(gyroy)
                inst.gyro_z.append(gyroz)
            else:
                inst.sample_posix.append(sample_posix)
                inst.sample_dnum.append(sample_dnum)
                inst.mag_x.append(magx)
                inst.mag_y.append(magy)
                inst.mag_z.append(magz)
                inst.accel_x.append(accelx)
                inst.accel_y.append(accely)
                inst.accel_z.append(accelz)
                inst.gyro_x.append(gyrox)
                inst.gyro_y.append(gyroy)
                inst.gyro_z.append(gyroz)

            i = star_idx + 3
            while i < n and payload[i] in (0x0D, 0x0A):
                i += 1

    def _parse_ecop_record(self, rec: Record, inst: ECOPData):
        inst._append_record_time(rec.posix, rec.dnum)
        if inst.file_mode:
            inst.raw_payloads.append(rec.payload)  # type: ignore[attr-defined]
        else:
            inst.raw_payloads.append(rec.payload)  # type: ignore[attr-defined]

    def _parse_som3_record(self, rec: Record, inst: SOM3Data):
        inst._append_record_time(rec.posix, rec.dnum)
        if inst.file_mode:
            inst.raw_payloads.append(rec.payload)  # type: ignore[attr-defined]
        else:
            inst.raw_payloads.append(rec.payload)  # type: ignore[attr-defined]


class DataProcessingThread(threading.Thread):
    def __init__(self, instruments: Dict[str, BaseInstrumentData], processed: Dict[str, BaseInstrumentData],
                 stop_event: threading.Event, file_mode: bool):
        super().__init__(daemon=True)
        self.instruments = instruments
        self.processed = processed
        self.stop_event = stop_event
        self.file_mode = file_mode
        self.last_t = {"TTV1": None, "TTV2": None, "TTV3": None}  # type: ignore[assignment]

        # flag the GUI can poll to open windows
        self.ttv_ready_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            self._process_ttv()
            time.sleep(0.05)  # ~20 Hz processing; tune as desired

    def _get_arr(self, inst: TTVData, name: str) -> np.ndarray:
        v = getattr(inst, name)
        if inst.file_mode:
            return np.asarray(v, dtype=float)
        return v.get()

    def _append_proc(self, proc: TTVProcessedData, tag: str, t_new: np.ndarray, bv_new: np.ndarray, bu_new: np.ndarray, bz_new: np.ndarray):
        if t_new.size == 0:
            return
        if proc.file_mode:
            if tag == "TTV1":
                proc.ttv1_posix.extend(t_new.tolist())
                proc.beam1_vel.extend(bv_new.tolist())
                proc.velZ1.extend(bz_new.tolist())
                proc.velU1.extend(bu_new.tolist())
            elif tag == "TTV2":
                proc.ttv2_posix.extend(t_new.tolist())
                proc.beam2_vel.extend(bv_new.tolist())
                proc.velZ2.extend(bz_new.tolist())
                proc.velU2.extend(bu_new.tolist())
            else:
                proc.ttv3_posix.extend(t_new.tolist())
                proc.beam3_vel.extend(bv_new.tolist())
                proc.velZ3.extend(bz_new.tolist())
                proc.velU3.extend(bu_new.tolist())
        else:
            if tag == "TTV1":
                proc.ttv1_posix.extend(t_new)
                proc.beam1_vel.extend(bv_new)
                proc.velZ1.extend(bz_new.tolist())
                proc.velU1.extend(bu_new.tolist())
            elif tag == "TTV2":
                proc.ttv2_posix.extend(t_new)
                proc.beam2_vel.extend(bv_new)
                proc.velZ2.extend(bz_new.tolist())
                proc.velU2.extend(bu_new.tolist())
            else:
                proc.ttv3_posix.extend(t_new)
                proc.beam3_vel.extend(bv_new)
                proc.velZ3.extend(bz_new.tolist())
                proc.velU3.extend(bu_new.tolist())

    def _process_ttv(self):
        proc = self.processed.get("TTV")
        if not isinstance(proc, TTVProcessedData):
            return

        for tag in ("TTV1", "TTV2", "TTV3"):
            inst = self.instruments.get(tag)
            if not isinstance(inst, TTVData):
                continue

            t = inst.get_sample_posix()
            if t.size < 2:
                continue

            tof_up = self._get_arr(inst, "tof_up")
            tof_dn = self._get_arr(inst, "tof_down")
            dtof   = self._get_arr(inst, "dtof")
            if tof_up.size != t.size or tof_dn.size != t.size or dtof.size != t.size:
                continue

            lt = self.last_t[tag]
            if lt is None:
                idx0 = 0
            else:
                idx0 = np.searchsorted(t, lt, side="right")

            if idx0 >= t.size:
                continue

            t_new = t[idx0:]
            u = tof_up[idx0:]
            d = tof_dn[idx0:]
            dt = dtof[idx0:]

            # beam velocity formula (vectorized)
            bv = np.zeros_like(dt, dtype=np.float64)
            bz = np.zeros_like(dt, dtype=np.float64)
            bu = np.zeros_like(dt, dtype=np.float64)
            m = (u > 0) & (d > 0)
            bv[m] = (TTV_SPACE / 2.0) * dt[m] / (u[m] * d[m])
            #Check the sin/cos ofr Z1,Z2,Z3 and U1,U2,U3
            bz[m] = bv[m]*math.cos(TTV_ANGLE_VERT2HOR)
            bu[m] = bv[m]*math.sin(TTV_ANGLE_VERT2HOR)


            self._append_proc(proc, tag, t_new, bv, bz, bu)

            self.last_t[tag] = float(t_new[-1])

            # “ready” once we have some processed points
            self.ttv_ready_event.set()


# =============================================================================
# GUI helpers
# =============================================================================
def _psd_from_timeseries(t: np.ndarray, y: np.ndarray):
    if t.size < 16:
        return None, None
    dt = np.median(np.diff(t))
    if dt <= 0:
        return None, None
    fs = 1.0 / dt
    n = t.size
    w = np.hanning(n)
    w2_sum = (w**2).sum()
    y_d = y - y.mean()
    y_w = y_d * w
    Y = np.fft.rfft(y_w)
    psd = (np.abs(Y) ** 2) / (fs * w2_sum)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    psd[psd <= 0] = 1e-20
    return f, psd


def _slice_last_window(t: np.ndarray, window_seconds: float) -> np.ndarray:
    if t.size == 0:
        return np.array([], dtype=bool)
    t0 = t[-1] - window_seconds
    # t is monotonic (expected). Use searchsorted to avoid O(N) mask work in huge arrays (file mode).
    idx0 = np.searchsorted(t, t0, side="left")
    mask = np.zeros_like(t, dtype=bool)
    mask[idx0:] = True
    return mask


def _decimate_xy(x: np.ndarray, y: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    if x.size <= max_points:
        return x, y
    step = int(x.size // max_points) + 1
    return x[::step], y[::step]


# =============================================================================
# Windows with throttled PSD + decimation + constant-time live reads
# =============================================================================
class EFE4Window(QtWidgets.QMainWindow):
    def __init__(self, inst_data: EFE4Data, use_full_series: bool):
        super().__init__()
        self.inst_data = inst_data
        self.use_full_series = use_full_series
        self.window_seconds = 5.0
        self.setWindowTitle("EFE4 – realtime (FAST)")

        self.max_plot_points_live = 4000
        self.max_plot_points_file = 20000
        self.psd_interval_s = 1.0  # <= point (2): PSD throttled
        self._last_psd_t = 0.0

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        left_layout = QtWidgets.QVBoxLayout()
        right_layout = QtWidgets.QVBoxLayout()
        layout.addLayout(left_layout, 1)
        layout.addLayout(right_layout, 1)

        self.ts_curves: Dict[str, pg.PlotDataItem] = {}

        p_t = pg.PlotWidget()
        p_t.setLabel("left", "T (V)")
        p_t.setLabel("bottom", "time", "s")
        self.ts_curves["t1"] = p_t.plot([], [], pen=pg.mkPen(EFE4_COLORS["t1"]), name="t1")
        self.ts_curves["t2"] = p_t.plot([], [], pen=pg.mkPen(EFE4_COLORS["t2"]), name="t2")
        p_t.addLegend()
        left_layout.addWidget(p_t)

        p_s = pg.PlotWidget()
        p_s.setLabel("left", "Shear (V)")
        p_s.setLabel("bottom", "time", "s")
        self.ts_curves["s1"] = p_s.plot([], [], pen=pg.mkPen(EFE4_COLORS["s1"]), name="s1")
        self.ts_curves["s2"] = p_s.plot([], [], pen=pg.mkPen(EFE4_COLORS["s2"]), name="s2")
        p_s.addLegend()
        left_layout.addWidget(p_s)

        p_a = pg.PlotWidget()
        p_a.setLabel("left", "Accel (g)")
        p_a.setLabel("bottom", "time", "s")
        self.ts_curves["a1"] = p_a.plot([], [], pen=pg.mkPen(EFE4_COLORS["a1"]), name="a1")
        self.ts_curves["a2"] = p_a.plot([], [], pen=pg.mkPen(EFE4_COLORS["a2"]), name="a2")
        self.ts_curves["a3"] = p_a.plot([], [], pen=pg.mkPen(EFE4_COLORS["a3"]), name="a3")
        p_a.addLegend()
        left_layout.addWidget(p_a)

        self.sp_widget = pg.PlotWidget()
        self.sp_widget.setLabel("left", "PSD")
        self.sp_widget.setLabel("bottom", "f", "Hz")
        self.sp_widget.setLogMode(x=True, y=True)
        self.sp_widget.showGrid(x=True, y=True, alpha=0.25)
        self.sp_widget.addLegend()
        right_layout.addWidget(self.sp_widget)

        self.sp_curves: Dict[str, pg.PlotDataItem] = {}
        for ch in ["t1", "t2", "s1", "s2", "a1", "a2", "a3"]:
            self.sp_curves[ch] = self.sp_widget.plot([], [], pen=pg.mkPen(EFE4_COLORS[ch]), name=ch)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(100)  # time series refresh

    def _get_channel(self, ch: str) -> np.ndarray:
        v = getattr(self.inst_data, ch)
        if self.inst_data.file_mode:
            return np.asarray(v, dtype=float)
        return v.get()

    def update_plots(self):
        t = self.inst_data.get_sample_posix()
        if t.size < 8:
            return

        if self.use_full_series:
            mask = np.ones_like(t, dtype=bool)
        else:
            mask = _slice_last_window(t, self.window_seconds)

        if mask.sum() < 8:
            return

        t_w = t[mask]
        t_rel = t_w - t_w[0]

        max_pts = self.max_plot_points_file if self.use_full_series else self.max_plot_points_live

        # Time-series: decimate
        for ch in ["t1", "t2", "s1", "s2", "a1", "a2", "a3"]:
            y_all = self._get_channel(ch)
            if y_all.size != t.size:
                continue
            y = y_all[mask]
            if y.size < 8:
                continue
            xx, yy = _decimate_xy(t_rel, y, max_pts)  # <= point (3)
            self.ts_curves[ch].setData(xx, yy)

        # PSD: throttled (<= point (2))
        now = time.time()
        if (now - self._last_psd_t) < self.psd_interval_s:
            return
        self._last_psd_t = now

        # PSD uses non-decimated window (but limited by window size anyway)
        for ch in ["t1", "t2", "s1", "s2", "a1", "a2", "a3"]:
            y_all = self._get_channel(ch)
            if y_all.size != t.size:
                continue
            y = y_all[mask]
            if y.size < 16:
                continue
            f, psd = _psd_from_timeseries(t_w, y)
            if f is not None:
                self.sp_curves[ch].setData(f, psd)


class TTVWindow(QtWidgets.QMainWindow):
    def __init__(self, ttv_dict: Dict[str, TTVData], use_full_series: bool):
        super().__init__()
        self.ttv_dict = ttv_dict
        self.use_full_series = use_full_series
        self.window_seconds = 5.0
        self.setWindowTitle("TTV1/TTV2/TTV3 – realtime (FAST)")

        self.max_plot_points_live = 3000
        self.max_plot_points_file = 20000
        self.psd_interval_s = 1.0
        self._last_psd_t = 0.0

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        left_layout = QtWidgets.QVBoxLayout()
        right_layout = QtWidgets.QVBoxLayout()
        layout.addLayout(left_layout, 1)
        layout.addLayout(right_layout, 1)

        self.ch_groups = [
            "tof_down",
            "tof_up",
            "dtof",
            "errorcode",
            "upstream_adcpeak",
            "downstream_adcpeak",
        ]

        self.ts_curves: Dict[str, Dict[str, pg.PlotDataItem]] = {}

        for ch in self.ch_groups:
            p = pg.PlotWidget()
            p.setLabel("left", ch)
            p.setLabel("bottom", "time", "s")
            p.addLegend()
            self.ts_curves[ch] = {}
            for tag in ("TTV1", "TTV2", "TTV3"):
                if tag in self.ttv_dict:
                    color = TTV_TAG_COLORS.get(tag, (255, 255, 255))
                    self.ts_curves[ch][tag] = p.plot([], [], pen=pg.mkPen(color), name=tag)
            left_layout.addWidget(p)

        self.sp_widget = pg.PlotWidget()
        self.sp_widget.setLabel("left", "PSD")
        self.sp_widget.setLabel("bottom", "f", "Hz")
        self.sp_widget.setLogMode(x=True, y=True)
        self.sp_widget.showGrid(x=True, y=True, alpha=0.25)
        self.sp_widget.addLegend()
        right_layout.addWidget(self.sp_widget)

        self.sp_curves: Dict[Tuple[str, str], pg.PlotDataItem] = {}
        for ch in self.ch_groups:
            for tag in ("TTV1", "TTV2", "TTV3"):
                if tag in self.ttv_dict:
                    color = TTV_TAG_COLORS.get(tag, (255, 255, 255))
                    self.sp_curves[(tag, ch)] = self.sp_widget.plot(
                        [], [], pen=pg.mkPen(color), name=f"{tag}-{ch}"
                    )

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(150)

    def _get_arr(self, inst: TTVData, name: str) -> np.ndarray:
        v = getattr(inst, name)
        if inst.file_mode:
            return np.asarray(v, dtype=float)
        return v.get()

    def update_plots(self):
        # latest across instruments for consistent live window
        latest = None
        for inst in self.ttv_dict.values():
            t = inst.get_sample_posix()
            if t.size:
                latest = t[-1] if latest is None else max(latest, t[-1])
        if latest is None:
            return

        max_pts = self.max_plot_points_file if self.use_full_series else self.max_plot_points_live

        # Time-series
        for tag, inst in self.ttv_dict.items():
            t = inst.get_sample_posix()
            if t.size < 8:
                continue

            if self.use_full_series:
                mask = np.ones_like(t, dtype=bool)
            else:
                t0 = latest - self.window_seconds
                idx0 = np.searchsorted(t, t0, side="left")
                mask = np.zeros_like(t, dtype=bool)
                mask[idx0:] = True

            if mask.sum() < 8:
                continue

            t_w = t[mask]
            t_rel = t_w - t_w[0]

            for ch in self.ch_groups:
                y_all = self._get_arr(inst, ch)
                if y_all.size != t.size:
                    continue
                y = y_all[mask]
                if y.size < 8:
                    continue
                xx, yy = _decimate_xy(t_rel, y, max_pts)
                self.ts_curves[ch][tag].setData(xx, yy)

        # PSD throttled
        now = time.time()
        if (now - self._last_psd_t) < self.psd_interval_s:
            return
        self._last_psd_t = now

        for tag, inst in self.ttv_dict.items():
            t = inst.get_sample_posix()
            if t.size < 16:
                continue
            if self.use_full_series:
                mask = np.ones_like(t, dtype=bool)
            else:
                t0 = latest - self.window_seconds
                idx0 = np.searchsorted(t, t0, side="left")
                mask = np.zeros_like(t, dtype=bool)
                mask[idx0:] = True
            if mask.sum() < 16:
                continue

            t_w = t[mask]
            for ch in self.ch_groups:
                y_all = self._get_arr(inst, ch)
                if y_all.size != t.size:
                    continue
                y = y_all[mask]
                if y.size < 16:
                    continue
                f, psd = _psd_from_timeseries(t_w, y)
                if f is not None:
                    self.sp_curves[(tag, ch)].setData(f, psd)

class TTVProcessedWindow(QtWidgets.QMainWindow):
    def __init__(self, proc: TTVProcessedData, use_full_series: bool):
        super().__init__()
        self.proc = proc
        self.use_full_series = use_full_series
        self.window_seconds = 5.0
        self.setWindowTitle("TTV Processed – beam velocities")

        self.max_plot_points_live = 3000
        self.max_plot_points_file = 20000

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.p = pg.PlotWidget()
        self.p.setLabel("left", "beam velocity", "m/s")
        self.p.setLabel("bottom", "time", "s")
        self.p.addLegend()
        layout.addWidget(self.p)

        self.p1 = pg.PlotWidget()
        self.p1.setLabel("left", " velocity Z", "m/s")
        self.p1.setLabel("bottom", "time", "s")
        self.p1.addLegend()
        layout.addWidget(self.p1)

        self.p2 = pg.PlotWidget()
        self.p2.setLabel("left", " velocity U", "m/s")
        self.p2.setLabel("bottom", "time", "s")
        self.p2.addLegend()
        layout.addWidget(self.p2)

        self.c1 = self.p.plot([], [], pen=pg.mkPen((255,0,0)), name="beam1")
        self.c2 = self.p.plot([], [], pen=pg.mkPen((0,255,0)), name="beam2")
        self.c3 = self.p.plot([], [], pen=pg.mkPen((0,0,255)), name="beam3")

        self.c4 = self.p1.plot([], [], pen=pg.mkPen((255,0,0)), name="Z1")
        self.c5 = self.p1.plot([], [], pen=pg.mkPen((0,255,0)), name="Z2")
        self.c6 = self.p1.plot([], [], pen=pg.mkPen((0,0,255)), name="Z3")

        self.c7 = self.p2.plot([], [], pen=pg.mkPen((255,0,0)), name="U1")
        self.c8 = self.p2.plot([], [], pen=pg.mkPen((0,255,0)), name="U2")
        self.c9 = self.p2.plot([], [], pen=pg.mkPen((0,0,255)), name="U3")


        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(100)

    def _get(self, name: str) -> np.ndarray:
        v = getattr(self.proc, name)
        if self.proc.file_mode:
            return np.asarray(v, dtype=float)
        return v.get()

    def update_plots(self):
        max_pts = self.max_plot_points_file if self.use_full_series else self.max_plot_points_live

        # beam1
        t1 = self._get("ttv1_posix");
        y1 = self._get("beam1_vel")
        z1 = self._get("velZ1")
        u1 = self._get("velU1")

        if t1.size >= 2 and y1.size == t1.size:
            if self.use_full_series:
                mask = np.ones_like(t1, dtype=bool)
            else:
                mask = _slice_last_window(t1, self.window_seconds)
            tw = t1[mask]; yr = y1[mask];zr = z1[mask];ur = u1[mask]
            if tw.size >= 2:
                x = tw - tw[0]
                xx, yy = _decimate_xy(x, yr, max_pts)
                xx, zz = _decimate_xy(x, zr, max_pts)
                xx, uu = _decimate_xy(x, ur, max_pts)
                self.c1.setData(xx, yy)
                self.c4.setData(xx, zz)
                self.c7.setData(xx, uu)

        # beam2
        t2 = self._get("ttv2_posix"); y2 = self._get("beam2_vel")
        z2 = self._get("velZ2");u2 = self._get("velU2")
        if t2.size >= 2 and y2.size == t2.size:
            mask = np.ones_like(t2, dtype=bool) if self.use_full_series else _slice_last_window(t2, self.window_seconds)
            tw = t2[mask]; yr = y2[mask];zr = z2[mask];ur = u2[mask];
            if tw.size >= 2:
                x = tw - tw[0]
                xx, yy = _decimate_xy(x, yr, max_pts)
                xx, zz = _decimate_xy(x, zr, max_pts)
                xx, uu = _decimate_xy(x, ur, max_pts)
                self.c2.setData(xx, yy)
                self.c5.setData(xx, zz)
                self.c8.setData(xx, uu)

        # beam3
        t3 = self._get("ttv3_posix"); y3 = self._get("beam3_vel")
        z3 = self._get("velZ3");u3 = self._get("velU3")
        if t3.size >= 2 and y3.size == t3.size:
            mask = np.ones_like(t3, dtype=bool) if self.use_full_series else _slice_last_window(t3, self.window_seconds)
            tw = t3[mask]; yr = y3[mask]
            zr = z3[mask]; ur = u3[mask]
            if tw.size >= 2:
                x = tw - tw[0]
                xx, yy = _decimate_xy(x, yr, max_pts)
                xx, zz = _decimate_xy(x, zr, max_pts)
                xx, uu = _decimate_xy(x, ur, max_pts)
                self.c3.setData(xx, yy)
                self.c6.setData(xx, zz)
                self.c9.setData(xx, uu)

class VNAVWindow(QtWidgets.QMainWindow):
    def __init__(self, inst_data: VNAVData, use_full_series: bool):
        super().__init__()
        self.inst_data = inst_data
        self.use_full_series = use_full_series
        self.window_seconds = 5.0
        self.setWindowTitle("VNAV – realtime (FAST)")

        self.max_plot_points_live = 3000
        self.max_plot_points_file = 20000
        self.psd_interval_s = 1.0
        self._last_psd_t = 0.0

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        left_layout = QtWidgets.QVBoxLayout()
        right_layout = QtWidgets.QVBoxLayout()
        layout.addLayout(left_layout, 1)
        layout.addLayout(right_layout, 1)

        self.ts_curves: Dict[str, pg.PlotDataItem] = {}

        p_mag = pg.PlotWidget()
        p_mag.setLabel("left", "Mag")
        p_mag.setLabel("bottom", "time", "s")
        p_mag.addLegend()
        for ch in ["mag_x", "mag_y", "mag_z"]:
            self.ts_curves[ch] = p_mag.plot([], [], pen=pg.mkPen(VNAV_COLORS[ch]), name=ch)
        left_layout.addWidget(p_mag)

        p_acc = pg.PlotWidget()
        p_acc.setLabel("left", "Accel")
        p_acc.setLabel("bottom", "time", "s")
        p_acc.addLegend()
        for ch in ["accel_x", "accel_y", "accel_z"]:
            self.ts_curves[ch] = p_acc.plot([], [], pen=pg.mkPen(VNAV_COLORS[ch]), name=ch)
        left_layout.addWidget(p_acc)

        p_gyr = pg.PlotWidget()
        p_gyr.setLabel("left", "Gyro")
        p_gyr.setLabel("bottom", "time", "s")
        p_gyr.addLegend()
        for ch in ["gyro_x", "gyro_y", "gyro_z"]:
            self.ts_curves[ch] = p_gyr.plot([], [], pen=pg.mkPen(VNAV_COLORS[ch]), name=ch)
        left_layout.addWidget(p_gyr)

        self.sp_widget = pg.PlotWidget()
        self.sp_widget.setLabel("left", "PSD")
        self.sp_widget.setLabel("bottom", "f", "Hz")
        self.sp_widget.setLogMode(x=True, y=True)
        self.sp_widget.showGrid(x=True, y=True, alpha=0.25)
        self.sp_widget.addLegend()
        right_layout.addWidget(self.sp_widget)

        self.sp_curves: Dict[str, pg.PlotDataItem] = {}
        for ch in ["mag_x", "mag_y", "mag_z", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]:
            self.sp_curves[ch] = self.sp_widget.plot([], [], pen=pg.mkPen(VNAV_COLORS[ch]), name=ch)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(120)

    def _get_channel(self, ch: str) -> np.ndarray:
        v = getattr(self.inst_data, ch)
        if self.inst_data.file_mode:
            return np.asarray(v, dtype=float)
        return v.get()

    def update_plots(self):
        t = self.inst_data.get_sample_posix()
        if t.size < 8:
            return

        if self.use_full_series:
            mask = np.ones_like(t, dtype=bool)
        else:
            mask = _slice_last_window(t, self.window_seconds)

        if mask.sum() < 8:
            return

        t_w = t[mask]
        t_rel = t_w - t_w[0]

        max_pts = self.max_plot_points_file if self.use_full_series else self.max_plot_points_live

        for ch in ["mag_x", "mag_y", "mag_z", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]:
            y_all = self._get_channel(ch)
            if y_all.size != t.size:
                continue
            y = y_all[mask]
            if y.size < 8:
                continue
            xx, yy = _decimate_xy(t_rel, y, max_pts)
            self.ts_curves[ch].setData(xx, yy)

        now = time.time()
        if (now - self._last_psd_t) < self.psd_interval_s:
            return
        self._last_psd_t = now

        for ch in ["mag_x", "mag_y", "mag_z", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]:
            y_all = self._get_channel(ch)
            if y_all.size != t.size:
                continue
            y = y_all[mask]
            if y.size < 16:
                continue
            f, psd = _psd_from_timeseries(t_w, y)
            if f is not None:
                self.sp_curves[ch].setData(f, psd)


class SB49Window(QtWidgets.QMainWindow):
    def __init__(self, inst_data: SB49Data, use_full_series: bool):
        super().__init__()
        self.inst_data = inst_data
        self.use_full_series = use_full_series
        self.window_seconds = 10.0
        self.setWindowTitle("SB49 – realtime (FAST)")

        self.max_plot_points_live = 3000
        self.max_plot_points_file = 20000
        self.psd_interval_s = 1.0
        self._last_psd_t = 0.0

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        left_layout = QtWidgets.QVBoxLayout()
        right_layout = QtWidgets.QVBoxLayout()
        layout.addLayout(left_layout, 1)
        layout.addLayout(right_layout, 1)

        self.ts_curves: Dict[str, pg.PlotDataItem] = {}

        pT = pg.PlotWidget()
        pT.setLabel("left", "T", "°C")
        pT.setLabel("bottom", "time", "s")
        self.ts_curves["T"] = pT.plot([], [], pen=pg.mkPen(SB49_COLORS["T"]), name="T")
        left_layout.addWidget(pT)

        pP = pg.PlotWidget()
        pP.setLabel("left", "P", "dbar")
        pP.setLabel("bottom", "time", "s")
        self.ts_curves["P"] = pP.plot([], [], pen=pg.mkPen(SB49_COLORS["P"]), name="P")
        left_layout.addWidget(pP)

        pS = pg.PlotWidget()
        pS.setLabel("left", "S", "psu")
        pS.setLabel("bottom", "time", "s")
        self.ts_curves["S"] = pS.plot([], [], pen=pg.mkPen(SB49_COLORS["S"]), name="S")
        left_layout.addWidget(pS)

        self.sp_widget = pg.PlotWidget()
        self.sp_widget.setLabel("left", "PSD")
        self.sp_widget.setLabel("bottom", "f", "Hz")
        self.sp_widget.setLogMode(x=True, y=True)
        self.sp_widget.showGrid(x=True, y=True, alpha=0.25)
        self.sp_widget.addLegend()
        right_layout.addWidget(self.sp_widget)

        self.sp_curves: Dict[str, pg.PlotDataItem] = {}
        for ch in ["T", "P", "S"]:
            self.sp_curves[ch] = self.sp_widget.plot([], [], pen=pg.mkPen(SB49_COLORS[ch]), name=ch)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(200)

    def _get_channel(self, ch: str) -> np.ndarray:
        v = getattr(self.inst_data, ch)
        if self.inst_data.file_mode:
            return np.asarray(v, dtype=float)
        return v.get()

    def update_plots(self):
        t = self.inst_data.get_sample_posix()
        if t.size < 8:
            return

        if self.use_full_series:
            mask = np.ones_like(t, dtype=bool)
        else:
            mask = _slice_last_window(t, self.window_seconds)

        if mask.sum() < 8:
            return

        t_w = t[mask]
        t_rel = t_w - t_w[0]

        max_pts = self.max_plot_points_file if self.use_full_series else self.max_plot_points_live

        for ch in ["T", "P", "S"]:
            y_all = self._get_channel(ch)
            if y_all.size != t.size:
                continue
            y = y_all[mask]
            if y.size < 8:
                continue
            xx, yy = _decimate_xy(t_rel, y, max_pts)
            self.ts_curves[ch].setData(xx, yy)

        now = time.time()
        if (now - self._last_psd_t) < self.psd_interval_s:
            return
        self._last_psd_t = now

        for ch in ["T", "P", "S"]:
            y_all = self._get_channel(ch)
            if y_all.size != t.size:
                continue
            y = y_all[mask]
            if y.size < 16:
                continue
            f, psd = _psd_from_timeseries(t_w, y)
            if f is not None:
                self.sp_curves[ch].setData(f, psd)


# =============================================================================
# Window manager
# =============================================================================
class InstrumentWindowManager(QtCore.QObject):
    def __init__(self, record_processor: RecordProcessorThread,data_processor: DataProcessingThread, use_full_series: bool, parent=None):
        super().__init__(parent)
        self.record_processor = record_processor
        self.data_processor = data_processor
        self.use_full_series = use_full_series

        self.efe4_window: Optional[EFE4Window] = None
        self.ttv_window: Optional[TTVWindow] = None
        self.vnav_window: Optional[VNAVWindow] = None
        self.sb49_window: Optional[SB49Window] = None
        self.ttv_proc_window: Optional[TTVProcessedWindow] = None

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.check_instruments)
        self.timer.start(500)

    @QtCore.pyqtSlot()
    def check_instruments(self):
        insts = self.record_processor.instruments

        if self.efe4_window is None and "EFE4" in insts:
            efe = insts["EFE4"]
            if isinstance(efe, EFE4Data) and efe.get_sample_posix().size:
                self.efe4_window = EFE4Window(efe, self.use_full_series)
                self.efe4_window.show()

        if self.ttv_window is None:
            ttv_dict: Dict[str, TTVData] = {}
            for tag in ("TTV1", "TTV2", "TTV3"):
                d = insts.get(tag)
                if isinstance(d, TTVData) and d.get_sample_posix().size:
                    ttv_dict[tag] = d
            if ttv_dict:
                self.ttv_window = TTVWindow(ttv_dict, self.use_full_series)
                self.ttv_window.show()

        if self.vnav_window is None and "VNAV" in insts:
            vnav = insts["VNAV"]
            if isinstance(vnav, VNAVData) and vnav.get_sample_posix().size:
                self.vnav_window = VNAVWindow(vnav, self.use_full_series)
                self.vnav_window.show()

        if self.sb49_window is None and "SB49" in insts:
            sb49 = insts["SB49"]
            if isinstance(sb49, SB49Data) and sb49.get_sample_posix().size:
                self.sb49_window = SB49Window(sb49, self.use_full_series)
                self.sb49_window.show()

        proc = self.record_processor.process_data.get("TTV")
        if self.ttv_proc_window is None and isinstance(proc, TTVProcessedData):
            # optional: only open once processing has actually produced something
            # use the thread’s event:
            if self.data_processor.ttv_ready_event.is_set():
                self.ttv_proc_window = TTVProcessedWindow(proc, self.use_full_series)
                self.ttv_proc_window.show()


# =============================================================================
# Quit management
# =============================================================================
class GlobalQuitFilter(QtCore.QObject):
    def __init__(self, on_quit_cb, parent=None):
        super().__init__(parent)
        self.on_quit_cb = on_quit_cb

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.KeyPress:
            if event.key() == QtCore.Qt.Key_Q:
                self.on_quit_cb()
                return True
        return False


def parse_tcp_arg(s: str) -> Tuple[str, int]:
    if ":" not in s:
        raise ValueError("TCP must be HOST:PORT")
    host, port_s = s.rsplit(":", 1)
    return host.strip(), int(port_s)


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="MOD-SOM state-machine reader/parser with realtime plots (pyqtgraph) FAST+STABLE + TCP + SB49/DCAL"
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", "-f", help="Path to input file")
    group.add_argument("--serial", "-s", help="Serial port (e.g. /dev/tty.usbserial)")
    group.add_argument("--tcp", help="TCP input as HOST:PORT (client connect)")
    ap.add_argument("--baud", "-b", type=int, default=115200, help="Serial baudrate")

    args = ap.parse_args()

    app = QtWidgets.QApplication([])

    byte_queue: queue.Queue = queue.Queue()
    record_queue: queue.Queue = queue.Queue()
    process_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    file_mode = bool(args.file)

    # processed container already exists inside record_processor
    # make sure TTVProcessedData exists:
    # record_processor.process_data["TTV"] = TTVProcessedData(file_mode, cap_ttv)


    record_processor = RecordProcessorThread(record_queue, stop_event, file_mode=file_mode)
    record_processor.start()

    data_proc = DataProcessingThread(
        instruments=record_processor.instruments,
        processed=record_processor.process_data,
        stop_event=stop_event,
        file_mode=file_mode,
    )
    data_proc.start()


    parser = EpsiStateMachineParser(record_queue)
    parser_thread = ParserThread(byte_queue, parser, record_queue, stop_event)
    parser_thread.start()

    source_thread = None
    file_obj = None
    ser = None
    sock = None

    use_full_series = bool(args.file)

    # -------- source open --------
    if args.file:
        file_obj = open(args.file, "rb")
        source_thread = ByteSourceThread(file_obj, mode="file", out_queue=byte_queue, stop_event=stop_event)
        source_thread.start()

    elif args.serial:
        if serial is None:
            print("pyserial is not installed. Install with: pip install pyserial")
            return
        ser = serial.Serial(port=args.serial, baudrate=args.baud, timeout=0.1)
        source_thread = ByteSourceThread(
            ser,
            mode="serial",
            out_queue=byte_queue,
            stop_event=stop_event,
            start_command=b"som.start\r\n",
            stop_command=b"som.stop\r\n",
        )
        source_thread.start()

    elif args.tcp:
        host, port = parse_tcp_arg(args.tcp)
        sock = socket.create_connection((host, port), timeout=5.0)
        sock.settimeout(0.1)
        print(f"[Main] Connected TCP to {host}:{port}")
        source_thread = ByteSourceThread(sock, mode="tcp", out_queue=byte_queue, stop_event=stop_event)
        source_thread.start()

    # -------- plotting manager --------
    manager = InstrumentWindowManager(record_processor,data_proc, use_full_series)
    # -------- unified quit path --------
    def request_quit():
        QtCore.QTimer.singleShot(0, app.quit)

    def on_quit():
        stop_event.set()

        if ser is not None and getattr(ser, "is_open", False):
            try:
                print("[Main] Sending som.stop to serial...")
                ser.write(b"som.stop\r\n")
                ser.flush()
            except Exception as e:
                print(f"[Main] Warning: could not send stop command on quit: {e}")

        byte_queue.put(None)

        try:
            if source_thread is not None:
                source_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            parser_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            record_queue.put(None)
            record_processor.join(timeout=1.0)
        except Exception:
            pass
        try:
            data_proc.join(timeout=1.0)
        except Exception:
            pass

        if file_obj is not None:
            try:
                file_obj.close()
            except Exception:
                pass
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass

    quit_filter = GlobalQuitFilter(request_quit)
    app.installEventFilter(quit_filter)

    def _sigint_handler(sig, frame):
        print("\n[Main] Ctrl-C received -> quitting...")
        request_quit()

    signal.signal(signal.SIGINT, _sigint_handler)
    app.aboutToQuit.connect(on_quit)

    _sig_timer = QtCore.QTimer()
    _sig_timer.timeout.connect(lambda: None)
    _sig_timer.start(200)

    app.exec_()


if __name__ == "__main__":
    main()
