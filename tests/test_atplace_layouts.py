"""Geometry intake checks; no external solver or training dependencies."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from build_atplace_layouts import pack, validate_layout


class LayoutTests(unittest.TestCase):
    def test_touching_allowed_overlap_rejected(self):
        chips = [dict(origin_um=[0, 0], size_um=[5, 5]),
                 dict(origin_um=[5, 0], size_um=[5, 5])]
        self.assertTrue(validate_layout(chips, [10, 5]))
        chips[1]['origin_um'] = [4, 0]
        with self.assertRaises(ValueError):
            validate_layout(chips, [10, 5])

    def test_bounds_rejected(self):
        with self.assertRaises(ValueError):
            validate_layout([dict(origin_um=[1, 0], size_um=[10, 5])], [10, 5])

    def test_packing_deterministic_and_conserves_inputs(self):
        case = dict(case_id='test', interposer_size_m=[.01, .01], chiplets=[
            dict(name='a', width_m=.004, depth_m=.006, power_W=10),
            dict(name='b', width_m=.006, depth_m=.004, power_W=20)])
        result = pack(case)
        self.assertEqual(result, pack(case))
        self.assertEqual(result['total_power_W'], 30)
        self.assertFalse(result['ready_for_fem'])
        for chip in result['chiplets']:
            self.assertEqual(chip['size_um'][0]*chip['size_um'][1], 24000000)

    def test_impossible_not_rescaled(self):
        case = dict(case_id='test', interposer_size_m=[.001, .001], chiplets=[
            dict(name='a', width_m=.002, depth_m=.002, power_W=10)])
        with self.assertRaises(ValueError):
            pack(case, attempts=2)

    def test_local_public_layouts(self):
        root = Path(__file__).resolve().parents[1]
        manifest = root / 'data/atplace_layouts/manifest.json'
        if not manifest.exists():
            self.skipTest('Run import_atplace_cases.py and build_atplace_layouts.py first')
        entries = json.loads(manifest.read_text(encoding='utf-8'))
        self.assertEqual(len(entries), 10)
        for entry in entries:
            if entry['status'] != 'geometry_validated':
                continue
            name = entry['case_id']
            layout = json.loads((manifest.parent / f'{name}.json').read_text(encoding='utf-8'))
            source = json.loads((root / f'data/atplace_import/{name}.json').read_text(encoding='utf-8'))
            validate_layout(layout['chiplets'], layout['interposer_size_um'])
            original = {c['name']: c for c in source['chiplets']}
            self.assertEqual(len(layout['chiplets']), len(original))
            self.assertAlmostEqual(layout['total_power_W'], source['total_power_W'])
            for chip in layout['chiplets']:
                c = original[chip['name']]
                self.assertEqual(sorted(chip['size_um']), sorted([round(c['width_m']*1e6), round(c['depth_m']*1e6)]))
                self.assertEqual(chip['power_W'], c['power_W'])


if __name__ == '__main__':
    unittest.main()
