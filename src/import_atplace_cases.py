"""Audit public ATPlace inputs without executing its placement binaries.

Bookshelf dimensions are micrometres (Thermal.py converts to mm, then m).
Source .pl positions are preserved without claiming a legal optimized layout.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path


def import_case(root: Path, name: str, sizes: dict) -> dict:
    folder = root / 'cases' / name
    powers = {}
    for line in (folder / f'{name}.power').read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        key, value = line.split()
        if key in powers:
            raise ValueError(f'duplicate power: {key}')
        powers[key] = float(value)
    positions = {}
    for line in (folder / f'{name}.pl').read_text().splitlines():
        parts = line.split()
        if parts and parts[0] in powers:
            positions[parts[0]] = [float(parts[1]), float(parts[2])]
    chips = []
    for line in (folder / f'{name}.blocks').read_text().splitlines():
        if 'hardrectilinear' not in line:
            continue
        key = line.split()[0]
        points = [(float(x), float(y)) for x, y in re.findall(r'\(([-\d.eE+]+),\s*([-\d.eE+]+)\)', line)]
        if len(points) != 4:
            raise ValueError(f'unsupported polygon: {key}')
        width = max(p[0] for p in points) - min(p[0] for p in points)
        depth = max(p[1] for p in points) - min(p[1] for p in points)
        if width <= 0 or depth <= 0 or powers[key] < 0:
            raise ValueError(f'invalid dimensions/power: {key}')
        chips.append(dict(name=key, width_m=width*1e-6, depth_m=depth*1e-6,
                          power_W=powers[key], source_position_um=positions.get(key)))
    if len({c['name'] for c in chips}) != len(chips) or set(powers) != {c['name'] for c in chips}:
        raise ValueError(f'chip/power mismatch: {name}')
    zero_positions = all(c['source_position_um'] == [0., 0.] for c in chips)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(folder.iterdir()) if p.is_file()}
    return dict(case_id=name, interposer_size_m=[v*1e-6 for v in sizes[name]],
                chiplets=chips, total_power_W=sum(powers.values()),
                layout_status='all_zero_initial_positions' if zero_positions else 'requires_legality_and_coordinate_audit',
                ready_for_fem=False, source_sha256=hashes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('external/atplace_download/ATPlace_pub-main'))
    parser.add_argument('--out', type=Path, default=Path('data/atplace_import'))
    args = parser.parse_args()
    tree = ast.parse((args.root / 'reproduce.py').read_text(encoding='utf-8'))
    sizes = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CASE_INTERPOSER_SIZE' for t in node.targets):
            sizes = ast.literal_eval(node.value)
    if sizes is None:
        raise ValueError('missing official interposer dimensions')
    cases = [import_case(args.root, f'Case{i}', sizes) for i in range(1, 11)]
    args.out.mkdir(parents=True, exist_ok=True)
    for case in cases:
        (args.out / (case['case_id'] + '.json')).write_text(json.dumps(case, indent=2), encoding='utf-8')
    audit = dict(source='https://github.com/PKU-IDEA/ATPlace_pub',
                 archive_sha256=hashlib.sha256(Path('external/atplace_source.zip').read_bytes()).hexdigest()
                 if Path('external/atplace_source.zip').exists() else None,
                 cases=[{k:v for k,v in c.items() if k not in ('chiplets','source_sha256')}
                        | {'chiplet_count':len(c['chiplets'])} for c in cases])
    (args.out / 'manifest.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
