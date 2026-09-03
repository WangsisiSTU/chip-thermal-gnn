# Reproducible 3D baseline

The baseline was run from `.venv-3d` on Python 3.12 with the following key
versions: scikit-fem 12.0.2, PyTorch 2.13.0+cpu, PyG 2.8.0.post1, SciPy
1.18.1, NumPy 2.5.2, and pytest 9.1.1.  This host has no CUDA device, so the
reported accelerator-memory field is intentionally `N/A` and CPU process RSS
is reported instead.

Create the isolated environment and install through the operating system trust
store (do not disable certificate validation):

```powershell
python -m venv .venv-3d
.\.venv-3d\Scripts\python.exe -m pip install --use-feature=truststore -r requirements.txt
```

Run the real FEM and graph-data checks:

```powershell
.\.venv-3d\Scripts\python.exe -m pytest tests/test_fem_solver_3d.py tests/test_graph_dataset_3d.py -q
```

The small default dataset is useful for a smoke run.  The baseline results use
the full fixed-seed 300/30/30 dataset and separate output directories:

```powershell
.\.venv-3d\Scripts\python.exe src/generate_dataset_3d.py --config configs/data_config_3d_full.yaml --out_dir data/raw_3d_full
.\.venv-3d\Scripts\python.exe src/graph_dataset_3d.py --raw_dir data/raw_3d_full --out_dir data/processed_3d_full
```

Train and evaluate the primary GraphSAGE baseline.  The configuration fixes a
3-layer, 64-hidden-unit model and caps CPU execution at four threads to avoid
thread oversubscription during PyG scatter operations.

```powershell
.\.venv-3d\Scripts\python.exe src/train.py --model baseline --data_dir data/processed_3d_full --config configs/train_config_3d.yaml --out_dir outputs/3d
.\.venv-3d\Scripts\python.exe src/evaluate.py --data_dir data/processed_3d_full --ckpt_dir outputs/3d/checkpoints --out_dir outputs/3d --cpu_threads 4
.\.venv-3d\Scripts\python.exe src/compare_2d_3d.py --model baseline
```

The compact MeshGraphNet trial uses two message-passing steps and a 64-wide
latent state (`configs/train_config_3d_mgn.yaml`).  Its 12-epoch result is a
cost/learning-rate probe, not a fully tuned replacement for the GraphSAGE
baseline.

Primary artifacts:

- `outputs/3d/metrics/baseline_eval_summary.json`: MAE, peak error, per-regime
  OOD metrics, inference time, FEM time, and memory.
- `outputs/3d/metrics/comparison_2d_vs_3d_baseline.md`: matched 2D/3D table.
- `outputs/3d/metrics/meshgraphnet_eval_summary.json`: compact MGN trial.

The unified master document for the 2D and 3D pipelines is
`docs/technical_documentation.md`.  The FEM formulation and original 3D design
rationale remain in `docs/three_dimensional_report.md`; the executed 3D record
is in `docs/three_dimensional_run_report.md`.
