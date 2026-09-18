"""Generate project-derived geometric baselines; not ATPlace optimized results.

Integer-micrometre MaxRects packing, optional 90 degree rotation, zero clearance.
This checks geometry only: routing, bump placement and thermal BCs are not solved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def overlaps(a, b):
    x, z, w, d = a
    u, v, s, t = b
    return x < u+s and u < x+w and z < v+t and v < z+d


def validate_layout(chips, size):
    rectangles = []
    for chip in chips:
        r = (*chip['origin_um'], *chip['size_um'])
        x, z, w, d = r
        if min(x, z) < 0 or min(w, d) <= 0 or x+w > size[0] or z+d > size[1]:
            raise ValueError('chip outside interposer')
        if any(overlaps(r, other) for other in rectangles):
            raise ValueError('overlapping chiplets')
        rectangles.append(r)
    return True


def pack(case, attempts=200):
    size = [round(v*1e6) for v in case['interposer_size_m']]
    source = case['chiplets']
    rng = random.Random(3107)
    for attempt in range(attempts):
        order = list(source)
        if attempt == 0:
            order.sort(key=lambda c: c['width_m']*c['depth_m'], reverse=True)
        elif attempt == 1:
            order.sort(key=lambda c: max(c['width_m'], c['depth_m']), reverse=True)
        else:
            rng.shuffle(order)
        free = [(0, 0, *size)]
        placed = []
        for chip in order:
            w, d = round(chip['width_m']*1e6), round(chip['depth_m']*1e6)
            options = []
            for x, z, fw, fd in free:
                for cw, cd, rot in [(w, d, 0), (d, w, 90)]:
                    if cw <= fw and cd <= fd:
                        options.append((min(fw-cw, fd-cd), max(fw-cw, fd-cd), z, x, cw, cd, rot))
            if not options:
                break
            _, _, z, x, cw, cd, rot = min(options)
            occupied = (x, z, cw, cd)
            updated = []
            for fx, fz, fw, fd in free:
                if not overlaps(occupied, (fx, fz, fw, fd)):
                    updated.append((fx, fz, fw, fd))
                    continue
                if x > fx:
                    updated.append((fx, fz, x-fx, fd))
                if x+cw < fx+fw:
                    updated.append((x+cw, fz, fx+fw-x-cw, fd))
                if z > fz:
                    updated.append((fx, fz, fw, z-fz))
                if z+cd < fz+fd:
                    updated.append((fx, z+cd, fw, fz+fd-z-cd))
            updated = sorted(set(updated))
            free = [a for a in updated if not any(a != b and a[0] >= b[0] and a[1] >= b[1]
                    and a[0]+a[2] <= b[0]+b[2] and a[1]+a[3] <= b[1]+b[3] for b in updated)]
            placed.append(dict(name=chip['name'], origin_um=[x, z], size_um=[cw, cd],
                               rotation_deg=rot, power_W=chip['power_W']))
        if len(placed) == len(source):
            validate_layout(placed, size)
            assert {c['name'] for c in placed} == {c['name'] for c in source}
            return dict(case_id=case['case_id'], layout_id='derived_maxrects_v1',
                        provenance='project-generated; NOT official optimized placement',
                        axes='x,z in plane; y reserved for stack; origin is lower-left',
                        interposer_size_um=size, chiplets=sorted(placed, key=lambda c: c['name']),
                        geometric_checks_passed=True, ready_for_fem=False,
                        clearance_um=0, routing_validated=False, packing_attempt=attempt,
                        total_power_W=sum(c['power_W'] for c in placed))
    raise ValueError(f"No legal packing found for {case['case_id']}; no dimensions were changed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path('data/atplace_import'))
    parser.add_argument('--out', type=Path, default=Path('data/atplace_layouts'))
    args = parser.parse_args()
    results = []
    audit = []
    for i in range(1, 11):
        source = args.input / f'Case{i}.json'
        try:
            layout = pack(json.loads(source.read_text(encoding='utf-8')))
        except ValueError as exc:
            audit.append(dict(case_id=f'Case{i}', status='packing_not_found', reason=str(exc)))
            continue
        layout['input_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
        results.append(layout)
        audit.append(dict(case_id=f'Case{i}', status='geometry_validated', ready_for_fem=False))
    args.out.mkdir(parents=True, exist_ok=True)
    for layout in results:
        (args.out / f"{layout['case_id']}.json").write_text(json.dumps(layout, indent=2), encoding='utf-8')
        print(f"{layout['case_id']}: {len(layout['chiplets'])} chips, legal geometry, {layout['total_power_W']} W")
    (args.out / 'manifest.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    for entry in audit:
        if entry['status'] != 'geometry_validated':
            print(f"{entry['case_id']}: packing not found (not a proof of infeasibility)")


if __name__ == '__main__':
    main()
