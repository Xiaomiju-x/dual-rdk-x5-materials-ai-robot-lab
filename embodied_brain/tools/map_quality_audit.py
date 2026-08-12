#!/usr/bin/env python3
"""Audit a ROS trinary PGM map without third-party dependencies."""

import argparse
import json
from collections import deque
from pathlib import Path


def load_pgm(path: Path):
    with path.open('rb') as handle:
        magic = handle.readline().strip()

        def token_line():
            line = handle.readline()
            while line.startswith(b'#') or not line.strip():
                line = handle.readline()
            return line

        width, height = map(int, token_line().split())
        max_value = int(token_line())
        if magic == b'P5':
            data = list(handle.read(width * height))
        elif magic == b'P2':
            data = [int(value) for value in handle.read().split()]
        else:
            raise ValueError(f'unsupported PGM magic {magic!r}')
    if max_value != 255 or len(data) != width * height:
        raise ValueError('invalid 8-bit PGM payload')
    return width, height, data


def connected_sizes(mask, width, height):
    seen = bytearray(len(mask))
    sizes = []
    for start, active in enumerate(mask):
        if not active or seen[start]:
            continue
        seen[start] = 1
        queue = [start]
        size = 0
        while queue:
            index = queue.pop()
            size += 1
            y, x = divmod(index, width)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        neighbor = ny * width + nx
                        if mask[neighbor] and not seen[neighbor]:
                            seen[neighbor] = 1
                            queue.append(neighbor)
        sizes.append(size)
    return sizes


def parse_image_from_yaml(path: Path) -> Path:
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip().startswith('image:'):
            value = line.split(':', 1)[1].strip().strip('"\'')
            image = Path(value)
            return image if image.is_absolute() else path.parent / image
    raise ValueError('map YAML has no image field')


def verdict(name, ok, detail):
    return {'name': name, 'status': 'PASS' if ok else 'FAIL', 'detail': detail}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('map', help='ROS map YAML or PGM path')
    parser.add_argument('--out')
    args = parser.parse_args()
    source = Path(args.map).resolve()
    pgm = parse_image_from_yaml(source) if source.suffix.lower() in {'.yaml', '.yml'} else source
    width, height, pixels = load_pgm(pgm)
    total = len(pixels)
    occupied = [value < 65 for value in pixels]
    free = [value > 250 for value in pixels]
    unknown = [not occ and not clear for occ, clear in zip(occupied, free)]
    occupied_count = sum(occupied)
    free_count = sum(free)
    unknown_count = sum(unknown)
    occ_sizes = connected_sizes(occupied, width, height)
    free_sizes = connected_sizes(free, width, height)
    largest_occ_ratio = max(occ_sizes, default=0) / max(1, occupied_count)
    largest_free_ratio = max(free_sizes, default=0) / max(1, free_count)
    tiny_occ_ratio = sum(size for size in occ_sizes if size <= 3) / max(1, occupied_count)

    checks = [
        verdict('map_dimensions', width >= 50 and height >= 50, f'{width}x{height}'),
        verdict('known_area', unknown_count / total <= 0.75,
                f'unknown_ratio={unknown_count / total:.4f}'),
        verdict('free_area', free_count / total >= 0.15,
                f'free_ratio={free_count / total:.4f}'),
        verdict('occupied_area', occupied_count / total >= 0.005,
                f'occupied_ratio={occupied_count / total:.4f}'),
        verdict('occupied_fragmentation', len(occ_sizes) <= 250,
                f'components={len(occ_sizes)}'),
        verdict('dominant_wall_structure', largest_occ_ratio >= 0.15,
                f'largest_component_ratio={largest_occ_ratio:.4f}'),
        verdict('tiny_obstacle_noise', tiny_occ_ratio <= 0.10,
                f'tiny_component_pixel_ratio={tiny_occ_ratio:.4f}'),
        verdict('free_space_connectivity', largest_free_ratio >= 0.80,
                f'largest_free_component_ratio={largest_free_ratio:.4f}'),
    ]
    result = {
        'schema': 'xrd-map-quality-audit-v1',
        'map': str(source),
        'image': str(pgm.resolve()),
        'metrics': {
            'width': width,
            'height': height,
            'unknown_ratio': unknown_count / total,
            'free_ratio': free_count / total,
            'occupied_ratio': occupied_count / total,
            'occupied_components': len(occ_sizes),
            'largest_occupied_component_ratio': largest_occ_ratio,
            'tiny_occupied_component_pixel_ratio': tiny_occ_ratio,
            'largest_free_component_ratio': largest_free_ratio,
        },
        'checks': checks,
        'passed': all(item['status'] == 'PASS' for item in checks),
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    print(payload)
    if args.out:
        Path(args.out).write_text(payload + '\n', encoding='utf-8')
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
