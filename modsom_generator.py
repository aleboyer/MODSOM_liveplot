#!/usr/bin/env python3
import argparse
import math
import socket
import struct
import time
from dataclasses import dataclass
from typing import Optional, Protocol, Dict, Any

try:
    import serial  # type: ignore
except ImportError:
    serial = None


# -----------------------------
# Writer backend (shared)
# -----------------------------
class Writer(Protocol):
    def write(self, b: bytes) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


class FileWriter:
    def __init__(self, path: str):
        self.f = open(path, "wb")

    def write(self, b: bytes) -> None:
        self.f.write(b)

    def flush(self) -> None:
        self.f.flush()

    def close(self) -> None:
        try:
            self.f.flush()
        except Exception:
            pass
        self.f.close()


class SerialWriter:
    def __init__(self, port: str, baud: int):
        if serial is None:
            raise RuntimeError("pyserial not installed. pip install pyserial")
        self.ser = serial.Serial(port=port, baudrate=baud, timeout=0.1)

    def write(self, b: bytes) -> None:
        self.ser.write(b)

    def flush(self) -> None:
        if hasattr(self.ser, "flush"):
            self.ser.flush()

    def close(self) -> None:
        try:
            self.ser.close()
        except Exception:
            pass


class TCPClientWriter:
    """Connect to host:port and send bytes."""
    def __init__(self, host: str, port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))

    def write(self, b: bytes) -> None:
        self.sock.sendall(b)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass


class TCPServerWriter:
    """Listen on host:port, accept one client, then send bytes to it."""
    def __init__(self, host: str, port: int):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(1)
        print(f"[TCP] listening on {host}:{port} ...")
        self.sock, addr = self.server.accept()
        print(f"[TCP] client connected from {addr[0]}:{addr[1]}")

    def write(self, b: bytes) -> None:
        self.sock.sendall(b)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass
        try:
            self.server.close()
        except Exception:
            pass


# -----------------------------
# MODSOM framing helpers
# -----------------------------
def _xor_bytes(data: bytes) -> int:
    x = 0
    for bb in data:
        x ^= bb
    return x & 0xFF


def build_modsom_record(tag4: bytes, ts_ms: int, payload: bytes) -> bytes:
    """
    Record format:
      $TAG + 16hex(ts_ms) + 8hex(payload_size) + *CC + payload + *PP
    """
    if len(tag4) != 4:
        raise ValueError("tag must be exactly 4 ASCII bytes, e.g. b'EFE4'")

    ts_hex = f"{ts_ms:016x}".encode("ascii")
    size_hex = f"{len(payload):08x}".encode("ascii")

    header_wo_ck = b"$" + tag4 + ts_hex + size_hex
    cc = _xor_bytes(header_wo_ck)
    header = header_wo_ck + b"*" + f"{cc:02X}".encode("ascii")

    pp = _xor_bytes(payload)
    trailer = b"*" + f"{pp:02X}".encode("ascii")

    return header + payload + trailer


def now_ms() -> int:
    return int(time.time() * 1000.0)


# -----------------------------
# DCAL structure + helpers
# -----------------------------
@dataclass
class DCAL:
    serial_no: str
    temperature_date: str
    conductivity_date: str
    pressure_date: str
    pressure_sn: str
    pressure_range_psia: float

    # temperature
    TA0: float
    TA1: float
    TA2: float
    TA3: float
    TOFFSET: float

    # conductivity
    G: float
    H: float
    I: float
    J: float
    CPCOR: float
    CTCOR: float
    CSLOPE: float

    # pressure
    PA0: float
    PA1: float
    PA2: float
    PTCA0: float
    PTCA1: float
    PTCA2: float
    PTCB0: float
    PTCB1: float
    PTCB2: float
    PTEMPA0: float
    PTEMPA1: float
    PTEMPA2: float
    POFFSET: float


def dcal_from_example() -> DCAL:
    """
    Your provided example, mapped into a structured DCAL object.
    This is what you'll later reuse to generate $SBE records.
    """
    return DCAL(
        serial_no="0131",
        temperature_date="03-apr-22",
        conductivity_date="03-apr-22",
        pressure_date="01-mar-22",
        pressure_sn="10166",
        pressure_range_psia=1450.0,

        TA0=8.021074e-04,
        TA1=2.854267e-04,
        TA2=-2.479661e-06,
        TA3=2.099656e-07,
        TOFFSET=0.0,

        G=-9.984149e-01,
        H=1.382838e-01,
        I=-1.271138e-04,
        J=3.163691e-05,
        CPCOR=-9.57e-08,
        CTCOR=3.25e-06,
        CSLOPE=1.0,

        PA0=6.234520e-01,
        PA1=4.463881e-03,
        PA2=1.824090e-12,
        PTCA0=5.203688e+05,
        PTCA1=6.073326e+00,
        PTCA2=2.926396e-03,
        PTCB0=2.484987e+01,
        PTCB1=2.175000e-03,
        PTCB2=0.0,
        PTEMPA0=-6.100266e+01,
        PTEMPA1=5.310386e+01,
        PTEMPA2=-5.031106e-01,
        POFFSET=0.0,
    )


def dcal_to_payload(d: DCAL) -> bytes:
    """
    Build the DCAL payload text (ALWAYS CRLF line endings),
    from the structured DCAL object.
    """
    lines = [
        f"  SERIAL NO. {d.serial_no}",
        f"temperature:  {d.temperature_date}",
        f"    TA0 = {d.TA0:.6e}",
        f"    TA1 = {d.TA1:.6e}",
        f"    TA2 = {d.TA2:.6e}",
        f"    TA3 = {d.TA3:.6e}",
        f"    TOFFSET = {d.TOFFSET:.6e}",
        f"conductivity:  {d.conductivity_date}",
        f"    G = {d.G:.6e}",
        f"    H = {d.H:.6e}",
        f"    I = {d.I:.6e}",
        f"    J = {d.J:.6e}",
        f"    CPCOR = {d.CPCOR:.6e}",
        f"    CTCOR = {d.CTCOR:.6e}",
        f"    CSLOPE = {d.CSLOPE:.6e}",
        f"pressure S/N = {d.pressure_sn}, range = {d.pressure_range_psia:g} psia:  {d.pressure_date}",
        f"    PA0 = {d.PA0:.6e}",
        f"    PA1 = {d.PA1:.6e}",
        f"    PA2 = {d.PA2:.6e}",
        f"    PTCA0 = {d.PTCA0:.6e}",
        f"    PTCA1 = {d.PTCA1:.6e}",
        f"    PTCA2 = {d.PTCA2:.6e}",
        f"    PTCB0 = {d.PTCB0:.6e}",
        f"    PTCB1 = {d.PTCB1:.6e}",
        f"    PTCB2 = {d.PTCB2:.6e}",
        f"    PTEMPA0 = {d.PTEMPA0:.6e}",
        f"    PTEMPA1 = {d.PTEMPA1:.6e}",
        f"    PTEMPA2 = {d.PTEMPA2:.6e}",
        f"    POFFSET = {d.POFFSET:.6e}",
        "",  # final blank line like your example (optional)
    ]
    txt = "\r\n".join(lines)
    return txt.encode("ascii", errors="strict")


def send_dcal_record(writer: Writer, dcal: DCAL) -> None:
    payload = dcal_to_payload(dcal)
    ts_ms = now_ms()
    rec = build_modsom_record(b"DCAL", ts_ms, payload)
    writer.write(rec)
    writer.flush()
    print(f"[GEN] Sent DCAL: payload={len(payload)} bytes total={len(rec)} bytes ts_ms=0x{ts_ms:016x}")


# -----------------------------
# Payload generators (EFE4/TTV/VNAV)
# -----------------------------
def sine01(t: float, f_hz: float = 1.0) -> float:
    return 0.5 * (1.0 + math.sin(2.0 * math.pi * f_hz * t))


def counts24_from_unit(u01: float) -> int:
    u01 = max(0.0, min(1.0, u01))
    return int(u01 * (2**24 - 1))


def pack_u24_be(x: int) -> bytes:
    x = max(0, min(2**24 - 1, int(x)))
    return x.to_bytes(3, byteorder="big", signed=False)


def make_efe4_payload(t0_posix: float, fs: float, n_samples: int) -> bytes:
    out = bytearray()
    for k in range(n_samples):
        t = t0_posix + k / fs
        ts_ms = int(t * 1000.0)
        out += struct.pack("<Q", ts_ms)

        u_t1 = sine01(t, 1.0)
        u_t2 = sine01(t + 0.10, 1.0)
        u_s1 = sine01(t + 0.20, 1.0)
        u_s2 = sine01(t + 0.30, 1.0)
        u_a1 = sine01(t + 0.40, 1.0)
        u_a2 = sine01(t + 0.50, 1.0)
        u_a3 = sine01(t + 0.60, 1.0)

        for u in (u_t1, u_t2, u_s1, u_s2, u_a1, u_a2, u_a3):
            out += pack_u24_be(counts24_from_unit(u))

    return bytes(out)


def make_ttv_payload(t0_posix: float, fs: float, n_samples: int, phase: float = 0.0) -> bytes:
    out = bytearray()
    for k in range(n_samples):
        t = t0_posix + k / fs
        ts_ms = int(t * 1000.0)

        ts_hex16 = f"{ts_ms:016x}".encode("ascii")
        s = math.sin(2.0 * math.pi * 1.0 * (t + phase))
        tof_up = 1.0 + 0.5 * s
        tof_dn = 1.2 + 0.5 * math.sin(2.0 * math.pi * 1.0 * (t + phase + 0.2))
        dtof = tof_up - tof_dn

        err = int((k % 8))
        up_peak = int(1000 + 200 * (0.5 * (1 + s)))
        dn_peak = int(1100 + 200 * (0.5 * (1 + math.sin(2.0 * math.pi * 1.0 * (t + phase + 0.3)))))

        out += ts_hex16
        out += struct.pack(">f", float(tof_up))
        out += struct.pack(">f", float(tof_dn))
        out += struct.pack(">f", float(dtof))
        out += struct.pack("B", err)
        out += struct.pack(">H", up_peak & 0xFFFF)
        out += struct.pack(">H", dn_peak & 0xFFFF)
        out += b"\r\n"

    return bytes(out)


def make_vnav_payload(t0_posix: float, fs: float, n_samples: int) -> bytes:
    out = bytearray()
    for k in range(n_samples):
        t = t0_posix + k / fs
        ts_ms = int(t * 1000.0)
        ts_hex16 = f"{ts_ms:016x}"

        magx = -0.2 + 0.5 * math.sin(2 * math.pi * 1.0 * (t + 0.00))
        magy = +0.0 + 0.5 * math.sin(2 * math.pi * 1.0 * (t + 0.10))
        magz = +0.2 + 0.5 * math.sin(2 * math.pi * 1.0 * (t + 0.20))
        accx = +9.7 + 0.2 * math.sin(2 * math.pi * 1.0 * (t + 0.30))
        accy = -1.2 + 0.2 * math.sin(2 * math.pi * 1.0 * (t + 0.40))
        accz = -0.2 + 0.2 * math.sin(2 * math.pi * 1.0 * (t + 0.50))
        gyx = -0.0003 + 0.0002 * math.sin(2 * math.pi * 1.0 * (t + 0.60))
        gyy = -0.0001 + 0.0002 * math.sin(2 * math.pi * 1.0 * (t + 0.70))
        gyz = -0.0001 + 0.0002 * math.sin(2 * math.pi * 1.0 * (t + 0.80))

        body = f"VNMAR,{magx:+.4f},{magy:+.4f},{magz:+.4f},{accx:+.3f},{accy:+.3f},{accz:+.3f},{gyx:+.6f},{gyy:+.6f},{gyz:+.6f}"
        computed = _xor_bytes(body.encode("ascii"))
        line = f"{ts_hex16}${body}*{computed:02X}\r\n"
        out += line.encode("ascii")
    return bytes(out)


# -----------------------------
# Run modes
# -----------------------------
@dataclass
class GenConfig:
    duration_s: float
    chunk_s: float
    realtime: bool
    efe4_fs: float
    ttv_fs: float
    vnav_fs: float


def run_generator(writer: Writer, cfg: GenConfig) -> None:
    t_start = time.time()
    t_end = t_start + cfg.duration_s
    next_chunk_start = t_start

    while time.time() < t_end:
        chunk_start = next_chunk_start
        next_chunk_start = chunk_start + cfg.chunk_s
        t0_posix = chunk_start

        n_efe4 = max(1, int(round(cfg.efe4_fs * cfg.chunk_s)))
        efe4_payload = make_efe4_payload(t0_posix, cfg.efe4_fs, n_efe4)
        writer.write(build_modsom_record(b"EFE4", int(t0_posix * 1000.0), efe4_payload))

        n_ttv = max(1, int(round(cfg.ttv_fs * cfg.chunk_s)))
        for tag, ph in ((b"TTV1", 0.0), (b"TTV2", 0.2), (b"TTV3", 0.4)):
            payload = make_ttv_payload(t0_posix, cfg.ttv_fs, n_ttv, phase=ph)
            writer.write(build_modsom_record(tag, int(t0_posix * 1000.0), payload))

        n_vnav = max(1, int(round(cfg.vnav_fs * cfg.chunk_s)))
        vnav_payload = make_vnav_payload(t0_posix, cfg.vnav_fs, n_vnav)
        writer.write(build_modsom_record(b"VNAV", int(t0_posix * 1000.0), vnav_payload))

        writer.flush()

        if cfg.realtime:
            now = time.time()
            sleep_s = next_chunk_start - now
            if sleep_s > 0:
                time.sleep(sleep_s)

    writer.flush()


# -----------------------------
# CLI
# -----------------------------
def parse_hostport(s: str) -> tuple[str, int]:
    if ":" not in s:
        raise ValueError("Expected host:port")
    host, port_s = s.rsplit(":", 1)
    return host, int(port_s)


def main():
    ap = argparse.ArgumentParser(description="MODSOM generator (file/serial/tcp share same writer backend)")

    out = ap.add_mutually_exclusive_group(required=True)
    out.add_argument("--file", help="Write to a raw file, e.g. modsom_0.modraw")
    out.add_argument("--serial", help="Write to a serial port, e.g. /dev/tty.usbserial-XXXX")
    out.add_argument("--tcp", help="TCP client connect host:port, e.g. 127.0.0.1:9000")
    out.add_argument("--tcp-listen", dest="tcp_listen", help="TCP server listen host:port, e.g. 0.0.0.0:9000")

    ap.add_argument("--baud", type=int, default=115200, help="Serial baud rate (if --serial)")

    ap.add_argument("--duration", type=float, default=30.0, help="Duration to generate (seconds)")
    ap.add_argument("--chunk", type=float, default=0.25, help="Record chunk interval (seconds)")
    ap.add_argument("--realtime", action="store_true", help="Sleep to simulate realtime (serial/tcp typical)")

    ap.add_argument("--efe4-fs", type=float, default=32.0, help="EFE4 sample rate (Hz)")
    ap.add_argument("--ttv-fs", type=float, default=8.0, help="TTV sample rate (Hz)")
    ap.add_argument("--vnav-fs", type=float, default=10.0, help="VNAV sample rate (Hz)")

    ap.add_argument("--no-dcal", action="store_true", help="Do not send DCAL at stream start")

    args = ap.parse_args()

    cfg = GenConfig(
        duration_s=float(args.duration),
        chunk_s=float(args.chunk),
        realtime=bool(args.realtime),
        efe4_fs=float(args.efe4_fs),
        ttv_fs=float(args.ttv_fs),
        vnav_fs=float(args.vnav_fs),
    )

    writer: Optional[Writer] = None
    try:
        if args.file:
            writer = FileWriter(args.file)
            if not args.realtime:
                cfg.realtime = False
        elif args.serial:
            writer = SerialWriter(args.serial, args.baud)
            if not args.realtime:
                cfg.realtime = True
        elif args.tcp:
            host, port = parse_hostport(args.tcp)
            writer = TCPClientWriter(host, port)
            if not args.realtime:
                cfg.realtime = True
        elif args.tcp_listen:
            host, port = parse_hostport(args.tcp_listen)
            writer = TCPServerWriter(host, port)
            if not args.realtime:
                cfg.realtime = True
        else:
            ap.error("No output mode selected.")

        # ---- build DCAL structure once; use it for DCAL record + later $SBE ----
        dcal = dcal_from_example()

        # ---- SEND DCAL ONCE AT START ----
        if not args.no_dcal:
            send_dcal_record(writer, dcal)

        print(f"[GEN] starting: duration={cfg.duration_s}s chunk={cfg.chunk_s}s realtime={cfg.realtime}")
        print(f"[GEN] EFE4 fs={cfg.efe4_fs}Hz, TTV fs={cfg.ttv_fs}Hz, VNAV fs={cfg.vnav_fs}Hz")
        run_generator(writer, cfg)
        print("[GEN] done.")

    except KeyboardInterrupt:
        print("\n[GEN] interrupted.")
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
