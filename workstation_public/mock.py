"""Mock data generators for Day 2+ frontend dev. Replaced by pymycobot/cv2 on Day 5+."""
import math
import time
import random


_EVENT_KINDS = [
    ('bpu_detect', 'workstation', 'ai_brain', '/api/bpu_detect_b64', 7),
    ('vlm_query', 'workstation', 'car_brain', '/api/vlm_query', 33),
    ('joint_sync', 'workstation', 'car_brain', '/api/joint_sync', 4),
    ('ocr_read', 'car_brain', 'ai_brain', '/api/ocr_value', 6),
    ('recipe', 'workstation', 'ai_brain', '/api/recipe', 12),
    ('predict', 'car_brain', 'ai_brain', '/api/predict', 31),
    ('failure_lib', 'workstation', 'ai_brain', '/api/failure_library', 9),
]


def mock_joints(arm):
    t = time.time()
    phase = 0.0 if arm == 'arm01' else math.pi
    return {
        'arm': arm,
        'ts': t,
        'angles': [30 * math.sin(t * 0.3 + phase + i * 0.5) for i in range(6)],
        'gripper': 50 + 30 * math.sin(t * 0.5 + phase),
    }


def mock_status():
    return {
        'arm01': {'online': True, 'temp_c': 38 + random.random() * 2},
        'arm02': {'online': True, 'temp_c': 39 + random.random() * 2},
        'cam01_fps': 28 + random.random() * 4,
        'cam02_fps': 28 + random.random() * 4,
        'ai_brain_ms': 8 + random.random() * 4,
        'car_brain_ms': 12 + random.random() * 6,
    }


def mock_coop_events(n=12):
    out = []
    now = time.time()
    for i in range(n):
        kind, src, dst, ep, base_ms = random.choice(_EVENT_KINDS)
        out.append({
            'ts': now - random.uniform(0, 30),
            'kind': kind,
            'src': src,
            'dst': dst,
            'endpoint': ep,
            'rtt_ms': base_ms + random.uniform(-2, 6),
            'bytes': random.choice([320, 1024, 4096, 81920, 320]),
            'ok': random.random() > 0.03,
        })
    out.sort(key=lambda e: -e['ts'])
    return out


def mock_coop_throughput():
    now = time.time()
    return {
        'now': now,
        'arm_to_ai': [10 + 5*math.sin(now*0.4 + i*0.3) + random.uniform(-1.5, 1.5) for i in range(30)],
        'arm_to_car': [6 + 4*math.cos(now*0.3 + i*0.4) + random.uniform(-1.2, 1.2) for i in range(30)],
        'car_to_ai': [4 + 3*math.sin(now*0.5 + i*0.2) + random.uniform(-0.8, 0.8) for i in range(30)],
    }
