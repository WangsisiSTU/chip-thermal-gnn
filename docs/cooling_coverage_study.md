# 3D cooling-coverage study

This study isolates low top-convection performance from strict extrapolation.
It uses `configs/data_config_3d_cooling_coverage.yaml` and keeps all generated
samples separate from the original 3D baseline directories.

## Dataset protocol

The dataset contains 360 training, 60 validation, and 150 test cases on the
same 1,859-node tetrahedral mesh. Every split has an independent random stream,
so changing a training count does not alter validation or test cases.

The normal operating interval is `h_top = 500..15000`. The supported
poor-cooling interval is `h_top = 100..400` and has 100/20/30 train/validation/
test cases. Strict poor-cooling extrapolation is kept separate at `h_top =
20..80`, with 25 test cases. High-power and poor-TIM strict OOD regimes also
have 25 test cases each.

Node features include the original local and global fields plus normalized
`log(h_top)` and an approximate fraction of die-centre path resistance lying on
the top-side path. This produces 26 node features. New checkpoints save the
feature names, feature normalization constants, and output normalization; the
evaluation and visualization commands reject a mismatch.

## First optimized GraphSAGE run

The baseline used three SAGE layers with 64 hidden channels, the global-maximum
peak loss, and the cooling-coverage training configuration. It ran for 150
epochs and selected epoch 141 by validation loss.

| Test regime | Samples | MAE (K) | Relative L2 | Mean peak error (K) |
| --- | ---: | ---: | ---: | ---: |
| ID | 45 | 0.0180 | 0.1080 | 0.0669 |
| Covered poor cooling | 30 | 0.0463 | 0.1096 | 0.1607 |
| Strict OOD: high power | 25 | 0.0730 | 0.1949 | 0.2972 |
| Strict OOD: poor TIM | 25 | 0.1105 | 0.5728 | 0.2660 |
| Strict OOD: poor cooling | 25 | 0.1565 | 0.2066 | 0.4086 |
| Overall | 150 | 0.0713 | 0.2810 | 0.2142 |

With three warmed forwards per sample and batches of four graphs, single-sample
inference was 6.70 ms and batch-amortized inference was 6.23 ms/sample. The
mean FEM solve time for this run was 20.29 ms, corresponding to 3.03x and 3.26x
speedups. Timing excludes first-forward dispatch costs.

These results are not directly comparable to the former 30-sample baseline:
the test size, regime proportions, strict poor-cooling interval, feature set,
and loss all changed. The useful conclusion is narrower: the supported
100..400 poor-cooling range has low error, while the unobserved 20..80 range
remains the dominant error source.

## Reproduction

```powershell
.\.venv-3d\Scripts\python.exe src/generate_dataset_3d.py `
  --config configs/data_config_3d_cooling_coverage.yaml `
  --out_dir data/raw_3d_cooling_coverage
.\.venv-3d\Scripts\python.exe src/graph_dataset_3d.py `
  --raw_dir data/raw_3d_cooling_coverage `
  --out_dir data/processed_3d_cooling_coverage
.\.venv-3d\Scripts\python.exe src/train.py --model baseline `
  --data_dir data/processed_3d_cooling_coverage `
  --config configs/train_config_3d_cooling_coverage.yaml `
  --out_dir outputs/3d/cooling_coverage --cpu_threads 4
.\.venv-3d\Scripts\python.exe src/evaluate.py `
  --data_dir data/processed_3d_cooling_coverage `
  --ckpt_dir outputs/3d/cooling_coverage/checkpoints `
  --out_dir outputs/3d/cooling_coverage --models baseline `
  --cpu_threads 4 --timing_repeats 3 --benchmark_batch_size 4
```