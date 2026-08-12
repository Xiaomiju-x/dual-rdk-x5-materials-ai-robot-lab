#!/usr/bin/env python3
"""探测串口雷达: 扫常见波特率, 抓裸帧, 按已知协议头识别型号."""
import sys, time, binascii

import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyUSB0'
BAUDS = (230400, 115200, 460800, 921600, 153600, 256000)

def identify(data: bytes) -> str:
    hits = []
    # LDROBOT LD06/LD14/LD19/D300: 0x54 0x2C 开头 47 字节包
    if b'\x54\x2c' in data:
        hits.append('LDROBOT (LD06/LD14/LD19/D300, 0x54 0x2C)')
    # Oradar MS200: 0x54 0xA5 帧头
    if b'\xa5\x54' in data or b'\x54\xa5' in data:
        hits.append('Oradar MS200 (0xA5 0x54)')
    # YDLIDAR: 0xAA 0x55
    if b'\xaa\x55' in data:
        hits.append('YDLIDAR family (0xAA 0x55)')
    # RPLIDAR: 0xA5 0x5A response descriptor
    if b'\xa5\x5a' in data:
        hits.append('RPLIDAR (0xA5 0x5A)')
    return ' / '.join(hits) if hits else 'UNKNOWN header'

for baud in BAUDS:
    try:
        s = serial.Serial(PORT, baud, timeout=1.5)
        s.reset_input_buffer()
        time.sleep(0.3)
        data = s.read(300)
        s.close()
        print(f'baud={baud} got={len(data)} bytes')
        if len(data) > 40:
            print('  hex:', binascii.hexlify(data[:94]).decode())
            print('  id :', identify(data))
            break
    except Exception as e:
        print(f'baud={baud} ERR {e}')
