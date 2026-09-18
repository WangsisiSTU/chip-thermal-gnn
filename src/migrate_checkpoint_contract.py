"""Attach a verified data contract to legacy project checkpoints."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from utils import attach_data_contract, load_json, validate_data_contract


def migrate(path: Path, metadata_path: Path) -> dict:
    metadata = load_json(str(metadata_path))
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('data_contract') is not None:
        validate_data_contract(checkpoint, metadata)
        return {'path': str(path), 'status': 'already_complete'}
    migrated = attach_data_contract(checkpoint, metadata)
    validate_data_contract(migrated, metadata)
    temporary = path.with_name(path.name + '.contract-tmp')
    torch.save(migrated, temporary)
    os.replace(temporary, path)
    return {
        'path': str(path),
        'status': 'migrated',
        'contract_fields': sorted(migrated['data_contract']),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoints', nargs='+', type=Path)
    parser.add_argument('--metadata', type=Path, required=True)
    args = parser.parse_args()
    results = [migrate(path, args.metadata) for path in args.checkpoints]
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
