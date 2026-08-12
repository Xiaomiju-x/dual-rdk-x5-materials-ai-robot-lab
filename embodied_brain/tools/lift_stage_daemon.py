#!/usr/bin/env python3
"""Persistent serial controller for the STM32F407 lift-stage firmware.

This keeps the CH340 serial port open so DTR/RTS transitions from repeated
short Python processes cannot reset the board or drop stepper enable.
Commands are written to a FIFO, one ASCII firmware command per line.
"""

import argparse
import os
import select
import signal
import sys
import time
from pathlib import Path

import serial


def log_line(log_fh, text: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp} {text}"
    print(line, flush=True)
    log_fh.write(line + "\n")
    log_fh.flush()


def ensure_fifo(path: Path) -> None:
    if path.exists():
        if not path.is_fifo():
            raise RuntimeError(f"{path} exists but is not a FIFO")
        return
    os.mkfifo(path, 0o666)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--fifo", default="/tmp/lift_stage_cmd.fifo")
    ap.add_argument("--log", default="/tmp/lift_stage_daemon.log")
    ap.add_argument("--pid", default="/tmp/lift_stage_daemon.pid")
    ap.add_argument("--status-period", type=float, default=1.0)
    args = ap.parse_args()

    fifo_path = Path(args.fifo)
    ensure_fifo(fifo_path)

    stop = False

    def handle_stop(_sig, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)

    with open(args.log, "a", buffering=1, encoding="utf-8") as log_fh:
        Path(args.pid).write_text(str(os.getpid()), encoding="ascii")
        log_line(log_fh, f"START port={args.port} fifo={args.fifo}")
        try:
            with serial.Serial(args.port, args.baud, timeout=0.02, dsrdtr=True) as ser:
                # Keep modem-control lines deasserted after open where adapters support it.
                try:
                    ser.dtr = False
                    ser.rts = False
                except Exception as exc:  # pragma: no cover - hardware dependent
                    log_line(log_fh, f"WARN modem-line-set failed: {exc!r}")

                fifo_rd = os.open(args.fifo, os.O_RDONLY | os.O_NONBLOCK)
                fifo_keep = os.open(args.fifo, os.O_WRONLY | os.O_NONBLOCK)
                buf = bytearray()
                pending = bytearray()
                last_status = 0.0

                def send(cmd: str) -> None:
                    cmd = cmd.strip()
                    if not cmd:
                        return
                    log_line(log_fh, f"> {cmd}")
                    ser.write((cmd + "\r\n").encode("ascii"))
                    ser.flush()

                send("PING")
                send("STATUS")

                while not stop:
                    now = time.time()
                    if now - last_status >= args.status_period:
                        send("STATUS")
                        last_status = now

                    try:
                        rlist, _, _ = select.select([fifo_rd], [], [], 0.05)
                    except InterruptedError:
                        continue
                    if fifo_rd in rlist:
                        data = os.read(fifo_rd, 4096)
                        if data:
                            pending.extend(data)
                            while b"\n" in pending:
                                raw, _, rest = pending.partition(b"\n")
                                pending[:] = rest
                                cmd = raw.decode("ascii", "replace").strip()
                                if cmd.upper() in {"QUIT", "EXIT"}:
                                    stop = True
                                    break
                                send(cmd)

                    data = ser.read(512)
                    if data:
                        buf.extend(data)
                        while b"\n" in buf:
                            raw, _, rest = buf.partition(b"\n")
                            buf[:] = rest
                            log_line(log_fh, "< " + raw.decode("ascii", "replace").rstrip())

                os.close(fifo_rd)
                os.close(fifo_keep)
        finally:
            try:
                Path(args.pid).unlink()
            except FileNotFoundError:
                pass
            log_line(log_fh, "STOP")
    return 0


if __name__ == "__main__":
    sys.exit(main())
