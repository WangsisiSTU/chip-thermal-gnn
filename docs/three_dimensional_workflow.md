# 3D tetrahedral workflow

The original 2D triangle-based workflow remains unchanged:

```powershell
python src/generate_dataset.py --config configs/data_config.yaml --out_dir data/raw
python src/graph_dataset.py --raw_dir data/raw --out_dir data/processed
```

The 3D workflow is independent and uses `y` as the package stack direction,
with `x` and `z` covering the die surface.  Its FEM mesh is P1 tetrahedral.
The heat source is a rectangular hot spot on the `x-z` plane that spans the
die thickness; its centre and widths are sampled independently in `x` and `z`.

## First run: small FEM smoke test

```powershell
python -m pytest tests/test_fem_solver_3d.py -q
# or inspect one small case directly
python src/fem_solver_3d.py --nx 5 --ny 13 --nz 5
```

The smoke test verifies tetrahedral connectivity, finite temperatures, the
bottom Dirichlet condition, top convection markers, and a positive die hot
spot response.  The default `ny=13` intentionally places all 1 mm-spaced layer
interfaces on mesh planes.

## Build the starter 3D dataset and PyG graphs

```powershell
python src/generate_dataset_3d.py --config configs/data_config_3d.yaml --out_dir data/raw_3d
python src/graph_dataset_3d.py --raw_dir data/raw_3d --out_dir data/processed_3d
```

Each raw mesh stores `points` with shape `(3, N)` and `tets` with shape `(4, M)`.
The processed PyG `Data` objects use `pos` with shape `(N, 3)`, 24 node features,
and 5 edge features (`dx`, `dy`, `dz`, length, harmonic-mean conductivity).
Tetrahedron edges are deduplicated and retained in both directions.

The generic training and evaluation commands support the 3D data directory
without code changes because they infer node and edge feature dimensions from
the dataset:

```powershell
python src/train.py --model meshgraphnet --data_dir data/processed_3d --out_dir outputs/3d
python src/train.py --model baseline --data_dir data/processed_3d --out_dir outputs/3d
python src/evaluate.py --data_dir data/processed_3d --ckpt_dir outputs/3d/checkpoints --out_dir outputs/3d
```

The starter 3D split is intentionally modest (24 / 4 / 6) for validation.  For
larger studies, increase `nx`, `ny`, `nz` and the split counts in
`configs/data_config_3d.yaml`; keep material interfaces aligned to y-grid planes
or use a mesh with explicitly tagged regions before drawing physical conclusions.
