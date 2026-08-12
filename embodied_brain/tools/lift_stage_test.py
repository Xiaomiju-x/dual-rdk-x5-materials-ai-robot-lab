#!/usr/bin/env python3
"""ASCII link test for the STM32F407 lift-stage firmware.

Run on the car X5 after flashing TEST_MODE=3 firmware:
    python3 lift_stage_test.py --port /dev/ttyUSB0 ping
    python3 lift_stage_test.py --port /dev/ttyUSB0 stepup
    python3 lift_stage_test.py --port /dev/ttyUSB0 up --steps 100 --pps 10
    python3 lift_stage_test.py --port /dev/ttyUSB0 cycle --steps 100 --pps 10
    python3 lift_stage_test.py --port /dev/ttyUSB0 mag on
    python3 lift_stage_test.py --port /dev/ttyUSB0 act ext --seconds 2
    python3 lift_stage_test.py --port /dev/ttyUSB0 servo left
    python3 lift_stage_test.py --port /dev/ttyUSB0 auto --steps 500 --pps 10
    python3 lift_stage_test.py --port /dev/ttyUSB0 demo --steps 50 --pps 10 --nav-ms 3000
    python3 lift_stage_test.py --port /dev/ttyUSB0 stop
    python3 lift_stage_test.py --port /dev/ttyUSB0 safezero --steps 500 --pps 10
    python3 lift_stage_test.py --port /dev/ttyUSB0 jogup
"""

import argparse
import sys
import time

import serial


def send_line(ser: serial.Serial, line: str) -> None:
    print(f"> {line}")
    ser.write((line.strip() + "\n").encode("ascii"))
    ser.flush()


def read_for(ser: serial.Serial, seconds: float) -> None:
    deadline = time.time() + seconds
    buf = bytearray()
    while time.time() < deadline:
        data = ser.read(256)
        if not data:
            continue
        buf.extend(data)
        while b"\n" in buf:
            line, _, rest = buf.partition(b"\n")
            buf[:] = rest
            print("< " + line.decode("ascii", "replace").rstrip())
    if buf:
        print("< " + buf.decode("ascii", "replace").rstrip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ping")
    sub.add_parser("status")
    sub.add_parser("stop")
    sub.add_parser("zero")

    p = sub.add_parser("safezero")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--pps", type=int, default=10)

    for name in ("up", "down", "cycle"):
        p = sub.add_parser(name)
        p.add_argument("--steps", type=int, default=25)
        p.add_argument("--pps", type=int, default=10)

    for name in ("jogup", "jogdown", "stepup", "stepdown"):
        p = sub.add_parser(name)
        p.add_argument("--steps", type=int, default=25)
        p.add_argument("--pps", type=int, default=10)

    p = sub.add_parser("goto")
    p.add_argument("target", type=int)
    p.add_argument("--pps", type=int, default=10)

    p = sub.add_parser("speed")
    p.add_argument("pps", type=int)
    p.add_argument("--seconds", type=float, default=3.0)

    p = sub.add_parser("mag")
    p.add_argument("state", choices=("on", "off"))

    p = sub.add_parser("act")
    p.add_argument("state", choices=("ext", "ret", "stop"))
    p.add_argument("--seconds", type=float, default=0.0)

    p = sub.add_parser("servo")
    p.add_argument("state", choices=("left", "home", "right", "us"))
    p.add_argument("--us", type=int, default=1500)

    p = sub.add_parser("auto")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--pps", type=int, default=10)
    p.add_argument("--extend-ms", type=int, default=6500)
    p.add_argument("--retract-ms", type=int, default=6500)
    p.add_argument("--hold-pps", type=int, default=0)
    p.add_argument("--seconds", type=float, default=0.0)

    p = sub.add_parser("demo")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--pps", type=int, default=10)
    p.add_argument("--nav-ms", type=int, default=3000)
    p.add_argument("--seconds", type=float, default=0.0)

    p = sub.add_parser("en")
    p.add_argument("value", type=int, choices=(0, 1))

    p = sub.add_parser("raw")
    p.add_argument("en_hi", type=int, choices=(0, 1))
    p.add_argument("dir_hi", type=int, choices=(0, 1))
    p.add_argument("pul_hi", type=int, choices=(0, 1))

    p = sub.add_parser("bitpulse")
    p.add_argument("en_hi", type=int, choices=(0, 1))
    p.add_argument("dir_hi", type=int, choices=(0, 1))
    p.add_argument("--pulses", type=int, default=200)
    p.add_argument("--half-ms", type=int, default=5)

    args = ap.parse_args()

    with serial.Serial(args.port, args.baud, timeout=0.05) as ser:
        read_for(ser, 0.5)
        if args.cmd == "ping":
            send_line(ser, "PING")
            read_for(ser, 1.0)
        elif args.cmd == "status":
            send_line(ser, "STATUS")
            read_for(ser, 1.0)
        elif args.cmd == "stop":
            send_line(ser, "STOP")
            read_for(ser, 1.0)
        elif args.cmd == "zero":
            send_line(ser, "ZERO")
            read_for(ser, 1.0)
        elif args.cmd == "safezero":
            send_line(ser, f"SAFEZERO {args.steps} {args.pps}")
            read_for(ser, max(3.0, args.steps / max(args.pps, 1) + 2.0))
        elif args.cmd == "en":
            send_line(ser, f"EN {args.value}")
            read_for(ser, 1.0)
        elif args.cmd == "raw":
            send_line(ser, f"RAW {args.en_hi} {args.dir_hi} {args.pul_hi}")
            read_for(ser, 1.0)
        elif args.cmd == "bitpulse":
            send_line(ser, f"BITPULSE {args.en_hi} {args.dir_hi} {args.pulses} {args.half_ms}")
            read_for(ser, max(3.0, args.pulses * max(args.half_ms, 1) * 2.0 / 1000.0 + 2.0))
        elif args.cmd == "goto":
            send_line(ser, f"GOTO {args.target} {args.pps}")
            read_for(ser, 4.0)
        elif args.cmd == "speed":
            send_line(ser, f"SPEED {args.pps}")
            read_for(ser, args.seconds)
        elif args.cmd == "mag":
            send_line(ser, f"MAG {args.state.upper()}")
            read_for(ser, 1.0)
        elif args.cmd == "act":
            send_line(ser, f"ACT {args.state.upper()}")
            read_for(ser, max(1.0, args.seconds))
            if args.state in ("ext", "ret") and args.seconds > 0:
                send_line(ser, "ACT STOP")
                read_for(ser, 1.0)
        elif args.cmd == "servo":
            if args.state == "us":
                send_line(ser, f"SERVO US {args.us}")
            else:
                send_line(ser, f"SERVO {args.state.upper()}")
            read_for(ser, 1.0)
        elif args.cmd == "auto":
            send_line(
                ser,
                f"AUTO {args.steps} {args.pps} {args.extend_ms} {args.retract_ms} {args.hold_pps}",
            )
            seconds = args.seconds
            if seconds <= 0:
                seconds = max(
                    10.0,
                    args.steps * 2.0 / max(args.pps, 1)
                    + (args.extend_ms + args.retract_ms) / 1000.0
                    + 8.0,
                )
            read_for(ser, seconds)
        elif args.cmd == "demo":
            send_line(ser, f"DEMO {args.steps} {args.pps} {args.nav_ms}")
            seconds = args.seconds
            if seconds <= 0:
                seconds = max(
                    12.0,
                    args.steps * 2.0 / max(args.pps, 1)
                    + args.nav_ms / 1000.0
                    + 10.0,
                )
            read_for(ser, seconds)
        elif args.cmd in ("jogup", "jogdown", "stepup", "stepdown"):
            send_line(ser, f"{args.cmd.upper()} {args.steps} {args.pps}")
            read_for(ser, max(4.0, args.steps / max(args.pps, 1) + 2.0))
        elif args.cmd in ("up", "down", "cycle"):
            send_line(ser, f"{args.cmd.upper()} {args.steps} {args.pps}")
            read_for(ser, max(3.0, args.steps / max(args.pps, 1) + 2.0))
        else:
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
