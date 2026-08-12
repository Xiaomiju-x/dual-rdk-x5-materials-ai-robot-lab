import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name('map_quality_audit.py')
SPEC = importlib.util.spec_from_file_location('map_quality_audit', MODULE_PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def write_pgm(path: Path, width: int, height: int, pixels) -> None:
    path.write_bytes(
        f'P5\n{width} {height}\n255\n'.encode('ascii') + bytes(pixels))


class MapQualityAuditTests(unittest.TestCase):
    def test_connected_room_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            width = height = 100
            pixels = [205] * (width * height)
            for y in range(10, 90):
                for x in range(10, 90):
                    pixels[y * width + x] = 254
            for x in range(10, 90):
                pixels[10 * width + x] = 0
                pixels[89 * width + x] = 0
            for y in range(10, 90):
                pixels[y * width + 10] = 0
                pixels[y * width + 89] = 0
            path = tmp_path / 'good.pgm'
            write_pgm(path, width, height, pixels)
            w, h, loaded = AUDIT.load_pgm(path)
            occupied = [value < 65 for value in loaded]
            free = [value > 250 for value in loaded]
            self.assertEqual((w, h), (100, 100))
            self.assertEqual(len(AUDIT.connected_sizes(occupied, w, h)), 1)
            self.assertEqual(len(AUDIT.connected_sizes(free, w, h)), 1)


    def test_fragmented_obstacles_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            width = height = 100
            pixels = [254] * (width * height)
            for y in range(2, 98, 3):
                for x in range(2, 98, 3):
                    pixels[y * width + x] = 0
            path = tmp_path / 'bad.pgm'
            write_pgm(path, width, height, pixels)
            w, h, loaded = AUDIT.load_pgm(path)
            occupied = [value < 65 for value in loaded]
            sizes = AUDIT.connected_sizes(occupied, w, h)
            self.assertGreater(len(sizes), 250)
            self.assertEqual(max(sizes), 1)


if __name__ == '__main__':
    unittest.main()
