#!/usr/bin/env python3
import argparse
import threading
import queue
import time
import struct
import signal
import socket
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from PyQt5 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg

try:
    import serial  # type: ignore
except ImportError:
    serial = None

VALID_TAGS = {
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

# ----------------------------------------------------------------------
# Time conversion
# ----------------------------------------------------------------------
MATLAB_EPOCH_DNUM = 719529.0  # datenum('1970-01-01')
SECONDS_PER_DAY = 86400.0


def posix_to_matlab_dnum(posix_seconds: float) -> float:
    return posix_seconds / SECONDS_PER_DAY + MATLAB_EPOCH_DNUM


# ----------------------------------------------------------------------
# ADC conversion helpers (3-byte signed, 24-bit)
# ----------------------------------------------------------------------
ADC_FULL_SCALE_COUNTS = 2 ** 24  # using 24-bit range for unipolar
ADC_FULL_SCALE_COUNTS_M1 = 2 ** 23  # for bipolar scaling
ADC_VREF_TEMP = 2.5
ADC_VREF_SHEAR = 2.5
ADC_VREF_ACCEL = 1.8
ACC_OFFSET = 1.8 / 2
ACC_FACTOR = 0.4


def bytes3_to_signed_int(b: bytes) -> int:
    """
    Convert 3 bytes (big-endian) to a 24-bit integer.
    (You can reintroduce sign handling here if needed.)
    """
    if len(b) != 3:
        raise ValueError("bytes3_to_signed_int expects exactly 3 bytes")
    raw = int.from_bytes(b, byteorder="big", signed=False)
    return raw


def counts24_to_volts_unipolar(counts: int, full_range: float) -> float:
    """
    Unipolar: FullRange * COUNTS / 2^24
    """
    return (counts / ADC_FULL_SCALE_COUNTS) * full_range


def counts24_to_volts_bipolar(counts: int, full_range: float) -> float:
    """
    Bipolar: FullRange * COUNTS / 2^(24-1)
    """
    return (counts / ADC_FULL_SCALE_COUNTS_M1) * full_range


def volts_to_g(acc_volts: float) -> float:
    """
    Convert accelerometer volts to g:
      g = (V - offset) / factor
    """
    return (acc_volts - ACC_OFFSET) / ACC_FACTOR


# ----------------------------------------------------------------------
# Color maps
# ----------------------------------------------------------------------
EFE4_COLORS = {
    "t1": (255, 0, 0),  # red
    "t2": (255, 165, 0),  # orange
    "s1": (0, 0, 255),  # blue
    "s2": (0, 255, 255),  # cyan
    "a1": (0, 255, 0),  # green
    "a2": (255, 0, 255),  # magenta
    "a3": (139, 69, 19),  # brown
}

TTV_TAG_COLORS = {
    "TTV1": (255, 0, 0),  # red
    "TTV2": (0, 255, 0),  # green
    "TTV3": (0, 0, 255),  # blue
}

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


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass
class Record:
    """Single record emitted by the parser."""
    inst_tag: str  # e.g. "EFE4", "TTV1"
    posix: float  # POSIX seconds since 1970-01-01 (header timestamp / 1000)
    dnum: float  # MATLAB datenum
    payload_size: int  # payload size in bytes
    payload: bytes  # raw payload bytes


@dataclass
class BaseInstrumentData:
    """
    Base instrument data:
      - record_* are timestamps at record level (per header)
      - sample_* are timestamps at sample level (inside each record)
    """
    record_posix: List[float] = field(default_factory=list)
    record_dnum: List[float] = field(default_factory=list)

    sample_posix: List[float] = field(default_factory=list)
    sample_dnum: List[float] = field(default_factory=list)


@dataclass
class EFE4Data(BaseInstrumentData):
    t1: List[float] = field(default_factory=list)
    t2: List[float] = field(default_factory=list)
    s1: List[float] = field(default_factory=list)
    s2: List[float] = field(default_factory=list)
    a1: List[float] = field(default_factory=list)
    a2: List[float] = field(default_factory=list)
    a3: List[float] = field(default_factory=list)
    raw_payloads: List[bytes] = field(default_factory=list)


@dataclass
class TTVData(BaseInstrumentData):
    tof_up: List[float] = field(default_factory=list)
    tof_down: List[float] = field(default_factory=list)
    dtof: List[float] = field(default_factory=list)
    errorcode: List[int] = field(default_factory=list)
    upstream_adcpeak: List[int] = field(default_factory=list)
    downstream_adcpeak: List[int] = field(default_factory=list)
    raw_payloads: List[bytes] = field(default_factory=list)


@dataclass
class SB49Data(BaseInstrumentData):
    raw_payloads: List[bytes] = field(default_factory=list)


@dataclass
class SB41Data(BaseInstrumentData):
    raw_payloads: List[bytes] = field(default_factory=list)


@dataclass
class VNAVData(BaseInstrumentData):
    mag_x: List[float] = field(default_factory=list)
    mag_y: List[float] = field(default_factory=list)
    mag_z: List[float] = field(default_factory=list)
    accel_x: List[float] = field(default_factory=list)
    accel_y: List[float] = field(default_factory=list)
    accel_z: List[float] = field(default_factory=list)
    gyro_x: List[float] = field(default_factory=list)
    gyro_y: List[float] = field(default_factory=list)
    gyro_z: List[float] = field(default_factory=list)
    raw_payloads: List[bytes] = field(default_factory=list)


@dataclass
class ECOPData(BaseInstrumentData):
    raw_payloads: List[bytes] = field(default_factory=list)


@dataclass
class SOM3Data(BaseInstrumentData):
    raw_payloads: List[bytes] = field(default_factory=list)


@dataclass
class ProcessedTTVData(BaseInstrumentData):
    """
    Processed TTV products (per-sample).
    Beam1 velocity:
      v = (L/2) * dtof / (tof_up * tof_down)
    Also store v*cos(theta) for comparison.
    """
    beam1: List[float] = field(default_factory=list)
    beam2: List[float] = field(default_factory=list)
    beam3: List[float] = field(default_factory=list)
    Xvel: List[float] = field(default_factory=list)
    Yvel: List[float] = field(default_factory=list)
    Z1vel: List[float] = field(default_factory=list)
    Z2vel: List[float] = field(default_factory=list)
    Z3vel: List[float] = field(default_factory=list)
    # optional: track which tag produced each sample (useful later for colors)
    src_tag: List[str] = field(default_factory=list)


@dataclass
class ProcessedEFEData(BaseInstrumentData):
    """Placeholder for processed EFE data."""
    pass


@dataclass
class ProcessedVNAVData(BaseInstrumentData):
    """Placeholder for processed VNAV data."""
    pass


# ----------------------------------------------------------------------
# Reader thread: file / serial / tcp socket
# ----------------------------------------------------------------------
class ByteSourceThread(threading.Thread):
    """
    Reads bytes from either:
      - file object  (read)
      - serial port  (read/write/flush)
      - TCP socket   (recv)
    and pushes chunks into out_queue.
    """

    def __init__(
            self,
            source,
            mode: str,  # "file" | "serial" | "tcp"
            out_queue: queue.Queue,
            stop_event: threading.Event,
            chunk_size: int = 4096,
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
        self.start_command = start_command
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
            # Serial: send start
            if self.mode == "serial" and self.start_command:
                try:
                    if self.clear_input_before_start and hasattr(self.source, "reset_input_buffer"):
                        self.source.reset_input_buffer()
                    self._send_serial_cmd(self.start_command)
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
            # Always try to send som.stop on serial if we are stopping
            if self.mode == "serial" and self.stop_command:
                self._send_serial_cmd(self.stop_command)

            # signal EOF to consumer
            self.out_queue.put(None)


# ----------------------------------------------------------------------
# Record processing thread
# ----------------------------------------------------------------------
class RecordProcessorThread(threading.Thread):
    def __init__(self, in_queue: queue.Queue, stop_event: threading.Event,
                 processed_queue: Optional[queue.Queue] = None):
        super().__init__(daemon=True)
        self.in_queue = in_queue
        self.stop_event = stop_event
        self.processed_queue = processed_queue

        self.instruments: Dict[str, BaseInstrumentData] = {
            "EFE4": EFE4Data(),
            "TTV1": TTVData(),
            "TTV2": TTVData(),
            "TTV3": TTVData(),
            "SB49": SB49Data(),
            "SB41": SB41Data(),
            "VNAV": VNAVData(),
            "ECOP": ECOPData(),
            "SOM3": SOM3Data(),
        }

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

        if tag == "EFE4":
            self._parse_efe4_record(rec, self.instruments["EFE4"])
        elif tag in ("TTV1", "TTV2", "TTV3"):
            self._parse_ttv_record(rec, self.instruments[tag], tag)
        elif tag == "SB49":
            self._parse_sb49_record(rec, self.instruments["SB49"])
        elif tag == "SB41":
            self._parse_sb41_record(rec, self.instruments["SB41"])
        elif tag == "VNAV":
            self._parse_vnav_record(rec, self.instruments["VNAV"])
        elif tag == "ECOP":
            self._parse_ecop_record(rec, self.instruments["ECOP"])
        elif tag == "SOM3":
            self._parse_som3_record(rec, self.instruments["SOM3"])

    # ---------------- Instrument parsers ------------------

    def _parse_efe4_record(self, rec: Record, inst: EFE4Data):
        inst.record_posix.append(rec.posix)
        inst.record_dnum.append(rec.dnum)
        inst.raw_payloads.append(rec.payload)

        payload = rec.payload
        SAMPLE_BYTES = 8 + 3 * 7
        n_samples = len(payload) // SAMPLE_BYTES

        if n_samples == 0:
            if len(payload) > 0:
                print(f"[EFE4] Warning: payload length {len(payload)} not a multiple of {SAMPLE_BYTES}")
            return

        if len(payload) % SAMPLE_BYTES != 0:
            print(
                f"[EFE4] Warning: payload length {len(payload)} not divisible by {SAMPLE_BYTES}. Using first {n_samples} samples.")

        for i in range(n_samples):
            offset = i * SAMPLE_BYTES
            sample_ts_bytes = payload[offset: offset + 8]
            chan_bytes = payload[offset + 8: offset + SAMPLE_BYTES]

            if len(sample_ts_bytes) != 8 or len(chan_bytes) != 3 * 7:
                print("[EFE4] Incomplete sample encountered, stopping.")
                break

            sample_ts_ms = int.from_bytes(sample_ts_bytes, byteorder="little", signed=False)
            sample_posix = sample_ts_ms / 1000.0
            sample_dnum = posix_to_matlab_dnum(sample_posix)
            inst.sample_posix.append(sample_posix)
            inst.sample_dnum.append(sample_dnum)

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
        inst.record_posix.append(rec.posix)
        inst.record_dnum.append(rec.dnum)
        inst.raw_payloads.append(rec.payload)

        payload = rec.payload
        PACKET_SIZE = 16 + 4 + 4 + 4 + 1 + 2 + 2 + 2  # 35 bytes (includes CRLF)

        if len(payload) < PACKET_SIZE:
            if len(payload) > 0:
                print(
                    f"[{tag}] Warning: payload length {len(payload)} smaller than one TTV sample ({PACKET_SIZE} bytes)")
            return

        n_packets = len(payload) // PACKET_SIZE
        if len(payload) % PACKET_SIZE != 0:
            print(
                f"[{tag}] Warning: payload length {len(payload)} not divisible by {PACKET_SIZE}. Using first {n_packets} samples.")

        for i in range(n_packets):
            offset = i * PACKET_SIZE
            chunk = payload[offset: offset + PACKET_SIZE]
            if len(chunk) < PACKET_SIZE:
                break

            ts_hex_bytes = chunk[0:16]
            try:
                ts_ms = int(ts_hex_bytes.decode("ascii", errors="ignore"), 16)
            except ValueError:
                print(f"[{tag}] Invalid hex timestamp in sample {i}: {ts_hex_bytes!r}")
                continue

            sample_posix = ts_ms / 1000.0
            sample_dnum = posix_to_matlab_dnum(sample_posix)
            inst.sample_posix.append(sample_posix)
            inst.sample_dnum.append(sample_dnum)

            idx = 16
            tof_up_bytes = chunk[idx: idx + 4]
            idx += 4
            tof_down_bytes = chunk[idx: idx + 4]
            idx += 4
            dtof_bytes = chunk[idx: idx + 4]
            idx += 4

            # NOTE: you used big-endian. If values look wrong, switch to "<f".
            tof_up = struct.unpack(">f", tof_up_bytes)[0]
            tof_down = struct.unpack(">f", tof_down_bytes)[0]
            dtof = struct.unpack(">f", dtof_bytes)[0]

            errorcode = chunk[idx]
            idx += 1

            upstream_adcpeak = struct.unpack(">H", chunk[idx: idx + 2])[0]
            idx += 2
            downstream_adcpeak = struct.unpack(">H", chunk[idx: idx + 2])[0]
            idx += 2

            crlf = chunk[idx: idx + 2]
            if crlf not in (b"\r\n", b"\n\r"):
                # tolerate
                pass

            inst.tof_up.append(tof_up)
            inst.tof_down.append(tof_down)
            inst.dtof.append(dtof)
            inst.errorcode.append(int(errorcode))
            inst.upstream_adcpeak.append(int(upstream_adcpeak))
            inst.downstream_adcpeak.append(int(downstream_adcpeak))

        if self.processed_queue is not None:
            # send the inst structure (already contains newly appended samples)
            self.processed_queue.put((tag, inst))

    def _parse_sb49_record(self, rec: Record, inst: SB49Data):
        inst.record_posix.append(rec.posix)
        inst.record_dnum.append(rec.dnum)
        inst.raw_payloads.append(rec.payload)

    def _parse_sb41_record(self, rec: Record, inst: SB41Data):
        inst.record_posix.append(rec.posix)
        inst.record_dnum.append(rec.dnum)
        inst.raw_payloads.append(rec.payload)

    def _parse_vnav_record(self, rec: Record, inst: VNAVData):
        inst.record_posix.append(rec.posix)
        inst.record_dnum.append(rec.dnum)
        inst.raw_payloads.append(rec.payload)

        payload = rec.payload
        n = len(payload)
        i = 0

        while i + 16 + 1 <= n:
            ts_bytes = payload[i: i + 16]
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

            body_for_cksum = payload[tag_pos + 1: star_idx]
            computed_cksum = 0
            for b in body_for_cksum:
                computed_cksum ^= b
            computed_cksum &= 0xFF

            cksum_bytes = payload[star_idx + 1: star_idx + 3]
            try:
                published_cksum = int(cksum_bytes.decode("ascii", errors="ignore"), 16)
            except ValueError:
                i = star_idx + 3
                continue

            if computed_cksum != published_cksum:
                print(
                    f"[VNAV] BAD sample checksum (computed=0x{computed_cksum:02X}, published=0x{published_cksum:02X})")

            msg_bytes = payload[tag_pos: star_idx]
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
            inst.sample_posix.append(sample_posix)
            inst.sample_dnum.append(sample_dnum)

            inst.mag_x.append(magx);
            inst.mag_y.append(magy);
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
        inst.record_posix.append(rec.posix)
        inst.record_dnum.append(rec.dnum)
        inst.raw_payloads.append(rec.payload)

    def _parse_som3_record(self, rec: Record, inst: SOM3Data):
        inst.record_posix.append(rec.posix)
        inst.record_dnum.append(rec.dnum)
        inst.raw_payloads.append(rec.payload)


class ProcessedProcessorThread(threading.Thread):
    """
    Consumes already-parsed instrument data structures (inst) and computes
    processed products incrementally (no re-parsing of raw records).

    For now:
      - processed TTV: Beam1 velocity from inst.dtof, inst.tof_up, inst.tof_down

    Queue messages:
      (tag: str, inst: BaseInstrumentData)  e.g. ("TTV1", TTVData())
    """

    def __init__(self, in_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.in_queue = in_queue
        self.stop_event = stop_event

        self.proc: Dict[str, BaseInstrumentData] = {
            "TTV": ProcessedTTVData(),  # combined output (all TTV1/2/3)
            "EFE": ProcessedEFEData(),
            "VNAV": ProcessedVNAVData(),
        }

        # constants
        self.L = 0.0382
        self.theta_deg = 90.0 - 53.0
        self.cos_theta = float(np.cos(np.deg2rad(self.theta_deg)))
        self.sin_theta = float(np.sin(np.deg2rad(self.theta_deg)))

        # keep track of where we last processed each incoming inst tag
        self._last_idx: Dict[str, int] = {}

    def run(self):
        while not self.stop_event.is_set():
            item = self.in_queue.get()
            if item is None:
                break
            try:
                tag, inst = item
            except Exception:
                continue

            if tag in ("TTV1", "TTV2", "TTV3") and isinstance(inst, TTVData):
                self._process_ttv_inst(tag, inst, self.proc["TTV"])
            elif tag == "EFE4" and isinstance(inst, EFE4Data):
                self._process_efe4_inst(tag, inst, self.proc["EFE"])
            elif tag == "VNAV" and isinstance(inst, VNAVData):
                self._process_vnav_inst(tag, inst, self.proc["VNAV"])

    def _process_ttv_inst(self, tag: str, inst: TTVData, out: ProcessedTTVData):
        """
        Incrementally consume new samples from inst (already parsed):
          inst.sample_posix, inst.sample_dnum, inst.tof_up, inst.tof_down, inst.dtof
        Compute:
          v = (L/2) * dtof / (tof_up * tof_down)
          vcos = v * cos(theta)
        Append into out.* lists.
        """

        # Determine how many samples are currently available (use timestamps as truth)
        n = len(inst.sample_posix)
        if n < 1:
            return

        # Find starting index for new samples for this tag
        i0 = self._last_idx.get(tag, 0)
        if i0 >= n:
            return

        # Defensive: ensure channel lists are at least length n
        # (they should be, if your parser always appends all fields together)
        if not (len(inst.sample_dnum) >= n and len(inst.tof_up) >= n and len(inst.tof_down) >= n and len(
                inst.dtof) >= n):
            # If there is a mismatch, only process up to the minimum safe length
            n_safe = min(len(inst.sample_posix), len(inst.sample_dnum),
                         len(inst.tof_up), len(inst.tof_down), len(inst.dtof))
            if i0 >= n_safe:
                return
            n = n_safe

        # Append record timestamps optionally (if you want)
        # out.record_posix.extend(inst.record_posix)  # usually not needed here
        # out.record_dnum.extend(inst.record_dnum)

        # Process only new samples [i0:n)
        for i in range(i0, n):
            tof_up = inst.tof_up[i]
            tof_dn = inst.tof_down[i]
            dtof = inst.dtof[i]

            denom = tof_up * tof_dn
            if denom == 0 or not np.isfinite(denom):
                continue

            v = (self.L / 2.0) * (dtof / denom)
            if not np.isfinite(v):
                continue

            out.sample_posix.append(inst.sample_posix[i])
            out.sample_dnum.append(inst.sample_dnum[i])
            out.beam1.append(float(v))
            out.Xvel.append(float(v * self.sin_theta))
            out.Z1vel.append(float(v * self.cos_theta))
            out.src_tag.append(tag)

        # Update last processed index for this tag
        self._last_idx[tag] = n

    def _process_efe4_inst(self, tag: str, inst: EFE4Data, out: ProcessedEFEData):
        """Placeholder for EFE processing."""
        pass

    def _process_vnav_inst(self, tag: str, inst: VNAVData, out: ProcessedVNAVData):
        """Placeholder for VNAV processing."""
        pass


# ----------------------------------------------------------------------
# Parser: state machine
# ----------------------------------------------------------------------
class EpsiStateMachineParser:
    """
    $TAG tttttttttttttttt AAAAAAAA *CC <payload> *PP
    """

    STATE_SYNC = 0
    STATE_TAG = 1
    STATE_HEADER = 2
    STATE_PAYLOAD = 3

    HEADER_LEN = 1 + 4 + 16 + 8 + 1 + 2

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

        header = self.buffer[:self.HEADER_LEN]

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
            del self.buffer[:self.HEADER_LEN]
            self.state = self.STATE_PAYLOAD
        else:
            print(
                f"[HEADER] tag={tag_str} BAD checksum (computed=0x{computed_cksum:02X}, published=0x{published_cksum:02X}), header={header!r}")
            del self.buffer[0]
            self.state = self.STATE_SYNC

        return True

    def _phase_payload(self) -> bool:
        if self.current_payload_size is None or self.current_tag is None or self.current_timestamp_ms is None:
            self.state = self.STATE_SYNC
            return True

        needed = self.current_payload_size + 1 + 2
        if len(self.buffer) < needed:
            return False

        payload = self.buffer[: self.current_payload_size]
        star_byte = self.buffer[self.current_payload_size]
        cksum_bytes = self.buffer[self.current_payload_size + 1: self.current_payload_size + 3]

        tag_str = self.current_tag.decode("ascii")

        if star_byte != ord("*"):
            print(f"[PAYLOAD] tag={tag_str} malformed payload (no '*'): {self.buffer[:needed]!r}")
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

        del self.buffer[:needed]
        self._reset_current()
        self.state = self.STATE_SYNC
        return True

    def _reset_current(self):
        self.current_tag = None
        self.current_timestamp_ms = None
        self.current_payload_size = None


# ----------------------------------------------------------------------
# Parser thread (consumes byte_queue, feeds state machine)
# ----------------------------------------------------------------------
class ParserThread(threading.Thread):
    def __init__(
            self,
            byte_queue: queue.Queue,
            parser: EpsiStateMachineParser,
            record_queue: queue.Queue,
            stop_event: threading.Event,
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


# ----------------------------------------------------------------------
# GUI helpers
# ----------------------------------------------------------------------
def _psd_from_timeseries(t: np.ndarray, y: np.ndarray):
    if t.size < 8:
        return None, None
    dt = np.median(np.diff(t))
    if dt <= 0:
        return None, None
    fs = 1.0 / dt
    n = t.size
    w = np.hanning(n)
    w2_sum = (w ** 2).sum()
    y_d = y - y.mean()
    y_w = y_d * w
    Y = np.fft.rfft(y_w)
    psd = (np.abs(Y) ** 2) / (fs * w2_sum)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    psd[psd <= 0] = 1e-20
    return f, psd


# NOTE: your window classes (EFE4Window/TTVWindow/VNAVWindow/InstrumentWindowManager)
# are unchanged from what you pasted (keep them as-is).
# To keep this message focused on TCP + quit/stop, I’m not duplicating them again.
#
# Paste your existing window classes below this line unchanged.
# ----------------------------------------------------------------------

class EFE4Window(QtWidgets.QMainWindow):
    """
    EFE4 window:
      Left:
        - t1,t2
        - s1,s2
        - a1,a2,a3
      Right:
        - Spectra of all 7 channels in one subplot + 24/20/16-bit noise floors.
    """

    def __init__(self, inst_data: EFE4Data, use_full_series: bool):
        super().__init__()
        self.inst_data = inst_data
        self.use_full_series = use_full_series
        self.window_seconds = 5.0  # used only if not full series

        self.setWindowTitle("EFE4 – realtime")

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        left_layout = QtWidgets.QVBoxLayout()
        right_layout = QtWidgets.QVBoxLayout()
        layout.addLayout(left_layout, 1)
        layout.addLayout(right_layout, 1)

        self.ts_curves: Dict[str, pg.PlotDataItem] = {}

        # 1) t1 & t2
        p_t = pg.PlotWidget()
        p_t.setLabel("left", "T (V)")
        p_t.setLabel("bottom", "time", "s")
        self.ts_curves["t1"] = p_t.plot([], [], pen=pg.mkPen(EFE4_COLORS["t1"]), name="t1")
        self.ts_curves["t2"] = p_t.plot([], [], pen=pg.mkPen(EFE4_COLORS["t2"]), name="t2")
        p_t.addLegend()
        left_layout.addWidget(p_t)

        # 2) s1 & s2
        p_s = pg.PlotWidget()
        p_s.setLabel("left", "Shear (V)")
        p_s.setLabel("bottom", "time", "s")
        self.ts_curves["s1"] = p_s.plot([], [], pen=pg.mkPen(EFE4_COLORS["s1"]), name="s1")
        self.ts_curves["s2"] = p_s.plot([], [], pen=pg.mkPen(EFE4_COLORS["s2"]), name="s2")
        p_s.addLegend()
        left_layout.addWidget(p_s)

        # 3) a1,a2,a3
        p_a = pg.PlotWidget()
        p_a.setLabel("left", "Accel (g)")
        p_a.setLabel("bottom", "time", "s")
        self.ts_curves["a1"] = p_a.plot([], [], pen=pg.mkPen(EFE4_COLORS["a1"]), name="a1")
        self.ts_curves["a2"] = p_a.plot([], [], pen=pg.mkPen(EFE4_COLORS["a2"]), name="a2")
        self.ts_curves["a3"] = p_a.plot([], [], pen=pg.mkPen(EFE4_COLORS["a3"]), name="a3")
        p_a.addLegend()
        left_layout.addWidget(p_a)

        # Spectral plot (single subplot)
        self.sp_widget = pg.PlotWidget()
        self.sp_widget.setLabel("left", "PSD")
        self.sp_widget.setLabel("bottom", "f", "Hz")
        self.sp_widget.setLogMode(x=True, y=True)
        self.sp_widget.addLegend()
        right_layout.addWidget(self.sp_widget)

        self.sp_curves: Dict[str, pg.PlotDataItem] = {}
        for ch in ["t1", "t2", "s1", "s2", "a1", "a2", "a3"]:
            self.sp_curves[ch] = self.sp_widget.plot(
                [], [], pen=pg.mkPen(EFE4_COLORS[ch]), name=ch
            )

        # Noise floors
        pen24 = pg.mkPen((150, 150, 150), style=QtCore.Qt.DashLine)
        pen20 = pg.mkPen((200, 100, 100), style=QtCore.Qt.DashLine)
        pen16 = pg.mkPen((100, 200, 100), style=QtCore.Qt.DashLine)
        self.noise24 = self.sp_widget.plot([], [], pen=pen24, name="24-bit NF")
        self.noise20 = self.sp_widget.plot([], [], pen=pen20, name="20-bit NF")
        self.noise16 = self.sp_widget.plot([], [], pen=pen16, name="16-bit NF")

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(100)

    def update_plots(self):
        if not self.inst_data.sample_posix:
            return

        t = np.asarray(self.inst_data.sample_posix, dtype=float)
        N = t.size
        if N < 2:
            return

        if self.use_full_series:
            mask = np.ones_like(t, dtype=bool)
        else:
            t0 = t[-1] - self.window_seconds
            mask = t >= t0

        if mask.sum() < 8:
            return

        t_w = t[mask]

        if self.use_full_series and t_w.size > 10000:
            step = t_w.size // 10000 + 1
            dec = np.zeros_like(mask, dtype=bool)
            idxs = np.where(mask)[0][::step]
            dec[idxs] = True
            mask = dec
            t_w = t[mask]

        t_rel = t_w - t_w[0]
        f_sample = None

        def _update_channel(ch_name):
            nonlocal f_sample
            if not hasattr(self.inst_data, ch_name):
                return None, None
            y_all = np.asarray(getattr(self.inst_data, ch_name), dtype=float)
            if y_all.size != t.size:
                return None, None
            y = y_all[mask]
            if y.size < 8:
                return None, None
            self.ts_curves[ch_name].setData(t_rel, y)
            f, psd = _psd_from_timeseries(t_w, y)
            if f is None:
                return None, None
            self.sp_curves[ch_name].setData(f, psd)
            f_sample = f
            return f, psd

        for ch in ["t1", "t2", "s1", "s2", "a1", "a2", "a3"]:
            _update_channel(ch)

        if f_sample is not None and f_sample.size > 0:
            f = f_sample
            nf24 = 1e-12 * np.ones_like(f)
            nf20 = 1e-10 * np.ones_like(f)
            nf16 = 1e-8 * np.ones_like(f)
            self.noise24.setData(f, nf24)
            self.noise20.setData(f, nf20)
            self.noise16.setData(f, nf16)


class TTVWindow(QtWidgets.QMainWindow):
    """
    TTV window: combines TTV1, TTV2, TTV3.
    """

    def __init__(self, ttv_dict: Dict[str, TTVData], use_full_series: bool):
        super().__init__()
        self.ttv_dict = ttv_dict
        self.use_full_series = use_full_series
        self.window_seconds = 5.0

        self.setWindowTitle("TTV1/TTV2/TTV3 – realtime")

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        left_layout = QtWidgets.QVBoxLayout()
        right_layout = QtWidgets.QVBoxLayout()
        layout.addLayout(left_layout, 1)
        layout.addLayout(right_layout, 1)

        ch_groups = [
            "tof_down",
            "tof_up",
            "dtof",
            "errorcode",
            "upstream_adcpeak",
            "downstream_adcpeak",
        ]

        self.ts_plots: Dict[str, pg.PlotWidget] = {}
        self.ts_curves: Dict[str, Dict[str, pg.PlotDataItem]] = {}

        for ch in ch_groups:
            p = pg.PlotWidget()
            p.setLabel("left", ch)
            p.setLabel("bottom", "time", "s")
            p.addLegend()
            self.ts_plots[ch] = p
            self.ts_curves[ch] = {}
            for tag in ("TTV1", "TTV2", "TTV3"):
                if tag in self.ttv_dict:
                    color = TTV_TAG_COLORS.get(tag, (255, 255, 255))
                    curve = p.plot([], [], pen=pg.mkPen(color), name=tag)
                    self.ts_curves[ch][tag] = curve
            left_layout.addWidget(p)

        self.sp_widget = pg.PlotWidget()
        self.sp_widget.setLabel("left", "PSD")
        self.sp_widget.setLabel("bottom", "f", "Hz")
        self.sp_widget.setLogMode(x=True, y=True)
        self.sp_widget.addLegend()
        right_layout.addWidget(self.sp_widget)

        self.sp_curves: Dict[tuple, pg.PlotDataItem] = {}
        for ch in ch_groups:
            for tag in ("TTV1", "TTV2", "TTV3"):
                if tag in self.ttv_dict:
                    name = f"{tag}-{ch}"
                    color = TTV_TAG_COLORS.get(tag, (255, 255, 255))
                    self.sp_curves[(tag, ch)] = self.sp_widget.plot(
                        [], [], pen=pg.mkPen(color), name=name
                    )

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(200)

    def update_plots(self):
        latest = None
        for tag, inst in self.ttv_dict.items():
            if inst.sample_posix:
                if latest is None or inst.sample_posix[-1] > latest:
                    latest = inst.sample_posix[-1]

        if latest is None:
            return

        for tag, inst in self.ttv_dict.items():
            if not inst.sample_posix:
                continue

            t = np.asarray(inst.sample_posix, dtype=float)
            N = t.size
            if N < 2:
                continue

            if self.use_full_series:
                mask = np.ones_like(t, dtype=bool)
            else:
                t0 = latest - self.window_seconds
                mask = t >= t0

            if mask.sum() < 8:
                continue

            t_w = t[mask]

            if self.use_full_series and t_w.size > 10000:
                step = t_w.size // 10000 + 1
                dec = np.zeros_like(mask, dtype=bool)
                idxs = np.where(mask)[0][::step]
                dec[idxs] = True
                mask = dec
                t_w = t[mask]

            t_rel = t_w - t_w[0]

            def _plot_ts_and_psd(ch_name: str):
                if not hasattr(inst, ch_name):
                    return
                y_all = np.asarray(getattr(inst, ch_name), dtype=float)
                if y_all.size != t.size:
                    return
                y = y_all[mask]
                if y.size < 8:
                    return
                if ch_name in self.ts_curves and tag in self.ts_curves[ch_name]:
                    self.ts_curves[ch_name][tag].setData(t_rel, y)
                f, psd = _psd_from_timeseries(t_w, y)
                if f is None:
                    return
                key = (tag, ch_name)
                if key in self.sp_curves:
                    self.sp_curves[key].setData(f, psd)

            for ch in [
                "tof_down",
                "tof_up",
                "dtof",
                "errorcode",
                "upstream_adcpeak",
                "downstream_adcpeak",
            ]:
                _plot_ts_and_psd(ch)


class VNAVWindow(QtWidgets.QMainWindow):
    """
    VNAV window:
      Left:
        - mag_x, mag_y, mag_z
        - accel_x, accel_y, accel_z
        - gyro_x, gyro_y, gyro_z
      Right:
        - spectral plot for all 9 channels
    """

    def __init__(self, inst_data: VNAVData, use_full_series: bool):
        super().__init__()
        self.inst_data = inst_data
        self.use_full_series = use_full_series
        self.window_seconds = 5.0

        self.setWindowTitle("VNAV – realtime")

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        left_layout = QtWidgets.QVBoxLayout()
        right_layout = QtWidgets.QVBoxLayout()
        layout.addLayout(left_layout, 1)
        layout.addLayout(right_layout, 1)

        self.ts_curves: Dict[str, pg.PlotDataItem] = {}

        # Mags
        p_mag = pg.PlotWidget()
        p_mag.setLabel("left", "Mag")
        p_mag.setLabel("bottom", "time", "s")
        p_mag.addLegend()
        for ch in ["mag_x", "mag_y", "mag_z"]:
            self.ts_curves[ch] = p_mag.plot([], [], pen=pg.mkPen(VNAV_COLORS[ch]), name=ch)
        left_layout.addWidget(p_mag)

        # Accels
        p_acc = pg.PlotWidget()
        p_acc.setLabel("left", "Accel")
        p_acc.setLabel("bottom", "time", "s")
        p_acc.addLegend()
        for ch in ["accel_x", "accel_y", "accel_z"]:
            self.ts_curves[ch] = p_acc.plot([], [], pen=pg.mkPen(VNAV_COLORS[ch]), name=ch)
        left_layout.addWidget(p_acc)

        # Gyros
        p_gyr = pg.PlotWidget()
        p_gyr.setLabel("left", "Gyro")
        p_gyr.setLabel("bottom", "time", "s")
        p_gyr.addLegend()
        for ch in ["gyro_x", "gyro_y", "gyro_z"]:
            self.ts_curves[ch] = p_gyr.plot([], [], pen=pg.mkPen(VNAV_COLORS[ch]), name=ch)
        left_layout.addWidget(p_gyr)

        # Spectral plot (single)
        self.sp_widget = pg.PlotWidget()
        self.sp_widget.setLabel("left", "PSD")
        self.sp_widget.setLabel("bottom", "f", "Hz")
        self.sp_widget.setLogMode(x=True, y=True)
        self.sp_widget.addLegend()
        right_layout.addWidget(self.sp_widget)

        self.sp_curves: Dict[str, pg.PlotDataItem] = {}
        for ch in [
            "mag_x", "mag_y", "mag_z",
            "accel_x", "accel_y", "accel_z",
            "gyro_x", "gyro_y", "gyro_z",
        ]:
            self.sp_curves[ch] = self.sp_widget.plot(
                [], [], pen=pg.mkPen(VNAV_COLORS[ch]), name=ch
            )

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(150)

    def update_plots(self):
        if not self.inst_data.sample_posix:
            return

        t = np.asarray(self.inst_data.sample_posix, dtype=float)
        N = t.size
        if N < 2:
            return

        if self.use_full_series:
            mask = np.ones_like(t, dtype=bool)
        else:
            t0 = t[-1] - self.window_seconds
            mask = t >= t0

        if mask.sum() < 8:
            return

        t_w = t[mask]

        if self.use_full_series and t_w.size > 10000:
            step = t_w.size // 10000 + 1
            dec = np.zeros_like(mask, dtype=bool)
            idxs = np.where(mask)[0][::step]
            dec[idxs] = True
            mask = dec
            t_w = t[mask]

        t_rel = t_w - t_w[0]

        for ch in [
            "mag_x", "mag_y", "mag_z",
            "accel_x", "accel_y", "accel_z",
            "gyro_x", "gyro_y", "gyro_z",
        ]:
            if not hasattr(self.inst_data, ch):
                continue
            y_all = np.asarray(getattr(self.inst_data, ch), dtype=float)
            if y_all.size != t.size:
                continue
            y = y_all[mask]
            if y.size < 8:
                continue

            self.ts_curves[ch].setData(t_rel, y)
            f, psd = _psd_from_timeseries(t_w, y)
            if f is None:
                continue
            self.sp_curves[ch].setData(f, psd)


class TTVProcessedWindow(QtWidgets.QMainWindow):
    """
    Window for Processed TTV data (Beam Velocity).
    """

    def __init__(self, processed_ttv: ProcessedTTVData, use_full_series: bool):
        super().__init__()
        self.data = processed_ttv
        self.use_full_series = use_full_series
        self.window_seconds = 10.0
        self.setWindowTitle("TTV Processed - Beam Velocity")

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.plot_vel = pg.PlotWidget()
        self.plot_vel.setLabel("left", "Velocity", "m/s")
        self.plot_vel.setLabel("bottom", "time", "s")
        self.plot_vel.addLegend()
        layout.addWidget(self.plot_vel)

        self.curves: Dict[str, pg.PlotDataItem] = {}
        for tag in ("TTV1", "TTV2", "TTV3"):
            color = TTV_TAG_COLORS.get(tag, (255, 255, 255))
            self.curves[f"{tag}_v"] = self.plot_vel.plot([], [], pen=pg.mkPen(color), name=f"{tag} Beam Vel")
            self.curves[f"{tag}_vcos"] = self.plot_vel.plot([], [], pen=pg.mkPen(color, style=QtCore.Qt.DashLine),
                                                            name=f"{tag} V*cos(theta)")

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(200)

    def update_plots(self):
        if not self.data.sample_posix:
            return

        t = np.asarray(self.data.sample_posix, dtype=float)
        v = np.asarray(self.data.beam1_vel, dtype=float)
        vcos = np.asarray(self.data.beam1_vel_cos, dtype=float)
        tags = np.asarray(self.data.src_tag)

        if t.size < 2:
            return

        latest = t[-1]
        if self.use_full_series:
            mask_time = np.ones_like(t, dtype=bool)
        else:
            mask_time = t >= (latest - self.window_seconds)

        for tag in ("TTV1", "TTV2", "TTV3"):
            mask_tag = (tags == tag)
            mask = mask_time & mask_tag
            if not np.any(mask):
                continue

            t_plot = t[mask] - t[0]
            self.curves[f"{tag}_v"].setData(t_plot, v[mask])
            self.curves[f"{tag}_vcos"].setData(t_plot, vcos[mask])


class InstrumentWindowManager(QtCore.QObject):
    """
    Opens one window for:
      - EFE4
      - TTV1/TTV2/TTV3 combined
      - VNAV
    as soon as there are samples.
    """

    def __init__(self, record_processor: RecordProcessorThread, use_full_series: bool, parent=None):
        super().__init__(parent)
        self.record_processor = record_processor
        self.use_full_series = use_full_series

        self.efe4_window: Optional[EFE4Window] = None
        self.ttv_window: Optional[TTVWindow] = None
        self.vnav_window: Optional[VNAVWindow] = None
        self.ttv_proc_window: Optional[TTVProcessedWindow] = None

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.check_instruments)
        self.timer.start(500)

    @QtCore.pyqtSlot()
    def check_instruments(self):
        insts = self.record_processor.instruments
        # Access processed data via a reference we'll need to pass or find
        # For this implementation, we assume processed_processor.proc is accessible
        # or we pass the processed_processor to the manager.
        # Updated main() will pass the processed_processor.

        # EFE4
        if self.efe4_window is None and "EFE4" in insts:
            efe = insts["EFE4"]
            if efe.sample_posix:
                self.efe4_window = EFE4Window(efe, self.use_full_series)
                self.efe4_window.show()

        # TTV1/2/3 combined
        if self.ttv_window is None:
            ttv_dict: Dict[str, TTVData] = {}
            for tag in ("TTV1", "TTV2", "TTV3"):
                if tag in insts and isinstance(insts[tag], TTVData) and insts[tag].sample_posix:
                    ttv_dict[tag] = insts[tag]
            if ttv_dict:
                self.ttv_window = TTVWindow(ttv_dict, self.use_full_series)
                self.ttv_window.show()

        # VNAV
        if self.vnav_window is None and "VNAV" in insts:
            vnav = insts["VNAV"]
            if vnav.sample_posix:
                self.vnav_window = VNAVWindow(vnav, self.use_full_series)
                self.vnav_window.show()

    def set_processed_processor(self, proc_thread: 'ProcessedProcessorThread'):
        self.proc_thread = proc_thread

    @QtCore.pyqtSlot()
    def check_processed(self):
        if not hasattr(self, 'proc_thread'):
            return

        proc_data = self.proc_thread.proc

        # TTV Processed
        if self.ttv_proc_window is None and "TTV" in proc_data:
            ttv_p = proc_data["TTV"]
            if ttv_p.sample_posix:
                self.ttv_proc_window = TTVProcessedWindow(ttv_p, self.use_full_series)
                self.ttv_proc_window.show()

        # Placeholders for other processed windows
        # if self.efe_proc_window is None and "EFE" in proc_data: ...


# ----------------------------------------------------------------------
# Quit management: allow 'q' key anywhere + Ctrl-C
# ----------------------------------------------------------------------
class GlobalQuitFilter(QtCore.QObject):
    def __init__(self, on_quit_cb, parent=None):
        super().__init__(parent)
        self.on_quit_cb = on_quit_cb

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.KeyPress:
            key = event.key()
            if key == QtCore.Qt.Key_Q:
                self.on_quit_cb()
                return True
        return False


def parse_tcp_arg(s: str) -> Tuple[str, int]:
    # Accept HOST:PORT
    if ":" not in s:
        raise ValueError("TCP must be HOST:PORT")
    host, port_s = s.rsplit(":", 1)
    return host.strip(), int(port_s)


# ----------------------------------------------------------------------
# Main driver: set up threads + Qt app
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="MOD-SOM state-machine reader/parser with realtime plots (pyqtgraph) + TCP"
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
    processed_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    processed_processor = ProcessedProcessorThread(processed_queue, stop_event)
    processed_processor.start()

    record_processor = RecordProcessorThread(record_queue, stop_event)
    record_processor.start()

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
        source_thread = ByteSourceThread(
            file_obj, mode="file", out_queue=byte_queue, stop_event=stop_event
        )
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
            # keep your default start/stop behavior:
            start_command=b"som.start\r\n",
            stop_command=b"som.stop\r\n",
        )
        source_thread.start()

    elif args.tcp:
        host, port = parse_tcp_arg(args.tcp)
        sock = socket.create_connection((host, port), timeout=2.0)
        sock.settimeout(0.1)  # non-blocking-ish loop in thread
        print(f"[Main] Connected TCP to {host}:{port}")
        source_thread = ByteSourceThread(
            sock, mode="tcp", out_queue=byte_queue, stop_event=stop_event
        )
        source_thread.start()

    # -------- plotting manager (unchanged) --------
    manager = InstrumentWindowManager(record_processor, use_full_series)
    manager.set_processed_processor(processed_processor)
    # Connect the check_processed to the timer or a new one
    manager.timer.timeout.connect(manager.check_processed)

    # -------- unified quit path --------
    def request_quit():
        # Safe to call multiple times
        QtCore.QTimer.singleShot(0, app.quit)

    def on_quit():
        stop_event.set()

        # If serial, explicitly send som.stop once here too (backup exists in thread.finally)
        if ser is not None and getattr(ser, "is_open", False):
            try:
                print("[Main] Sending som.stop to serial...")
                ser.write(b"som.stop\r\n")
                ser.flush()
            except Exception as e:
                print(f"[Main] Warning: could not send stop command on quit: {e}")

        # Unblock parser pipeline
        byte_queue.put(None)

        # Join threads
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
            processed_queue.put(None)
            processed_processor.join(timeout=1.0)
        except Exception:
            pass

        # Close resources
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

    # Press 'q' anywhere in the app
    quit_filter = GlobalQuitFilter(request_quit)
    app.installEventFilter(quit_filter)

    # Ctrl-C support: schedule app.quit inside Qt event loop
    def _sigint_handler(sig, frame):
        print("\n[Main] Ctrl-C received -> quitting...")
        request_quit()

    signal.signal(signal.SIGINT, _sigint_handler)

    app.aboutToQuit.connect(on_quit)

    # Small timer keeps the Qt event loop responsive to SIGINT on some platforms
    _sig_timer = QtCore.QTimer()
    _sig_timer.timeout.connect(lambda: None)
    _sig_timer.start(200)

    app.exec_()


if __name__ == "__main__":
    main()
