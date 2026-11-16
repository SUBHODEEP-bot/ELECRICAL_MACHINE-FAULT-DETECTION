# Electrical Machine Fault Detection

A dataset-agnostic baseline for machine-condition monitoring. The pipeline can
train three complementary models from one canonical sample table:

- Fault-family classification (Random Forest)
- Unsupervised anomaly detection (Isolation Forest trained on healthy windows)
- Remaining useful life (RUL) regression (Random Forest)

The PDF contains **12** dataset entries, not 11. They cover different machines,
fault taxonomies, and native file formats. Their catalogue is in
[`datasets.json`](datasets.json). The PDF itself is a list of sources, not the
sensor data.

## Trained reference models

These are separate, source-specific baselines; they are not one model trained
to recognize every machine and fault:

| Dataset | Task | Held-out result |
|---|---|---|
| CWRU, 12 kHz Drive End | Fault classification | Asset-group accuracy and macro-F1: 1.00 |
| XJTU-SY | RUL regression | MAE: about 3,420 seconds on a two-bearing holdout, versus about 4,719 seconds for a constant baseline |
| NASA FEMTO / PRONOSTIA | RUL regression | Full-test-bearing MAE: 4,903 seconds versus 6,219 seconds for a constant baseline; operating-condition-grouped CV did not beat baseline |
| Ottawa / UORED-VAFCLS | Fault classification and anomaly detection | Five-fold bearing-group classification accuracy 0.581 / macro-F1 0.544; anomaly F1 0.749 |
| MAFAULDA | Six-class fault classification and anomaly detection | Five-fold exact-speed-grouped classification accuracy 0.991 / macro-F1 0.983; anomaly F1 0.957 |

MAFAULDA's six classes are normal, horizontal misalignment, vertical
misalignment, imbalance, underhang bearing, and overhang bearing. The
classification holdout keeps all recordings with the same filename speed
group together; nearby speeds may still occur in different folds. These are
laboratory simulator results, not field validation.

The FEMTO model uses six complete training bearings and evaluates on the 11
complete `Full_Test_Set` bearings. Its target is time to the last recorded
capture (10-second capture spacing), not a guaranteed field failure time. The
separate operating-condition-grouped cross-validation MAE was about 6,792
seconds versus 6,041 seconds for a constant baseline, so results vary with the
split and the model is experimental rather than deployment-ready.

Ottawa scores are modest and should not be treated as field validation. No RUL
model was trained because the source stages do not provide a verified
time-to-failure interval.

The MAFAULDA anomaly detector was fit only on healthy training-fold windows.
Its exact-speed-grouped out-of-fold precision was 0.995, recall 0.921, and
F1 0.957. The six-class model's majority-class baseline accuracy was 0.286.

An IMS RUL experiment was not promoted: its held-out MAE was about 1,016,490
seconds, worse than the 691,847-second constant baseline. Its target is time to
the last recorded capture, not a confirmed physical failure time. Its holdout
also shares an acquisition with another bearing, so it is not independent
machine validation.

A SEUC gearbox experiment is also not suitable for use: its held-out
operating-condition result was 0.20 accuracy and 0.0668 macro-F1. The source's
bearing labels also conflict between file names and embedded titles, so no
SEUC bearing model was trained.

Several catalogue entries remain unavailable or unverified: the MFPT link no
longer exposes the data files, the listed Paderborn catalogue URL returns 404,
and the PHM 2009 gearbox and CWRU+DIRG sources were not located at their listed
URLs. FEMTO 2014 is a separate catalogue entry and has not been verified as
equivalent to FEMTO/PRONOSTIA.

## Algorithms

The shared pipeline uses:

- `RandomForestClassifier` for fault labels: 200 trees, minimum leaf size 2,
  and balanced subsample class weights.
- `RandomForestRegressor` for RUL: 200 trees and minimum leaf size 2.
- `IsolationForest` for anomaly detection: 200 trees, fit only on healthy
  windows.

Each vibration window is represented by statistical and spectral features:
mean, standard deviation, RMS, peak-to-peak, absolute mean, skew, kurtosis,
crest factor, dominant frequency, spectral centroid, and three band-power
ratios per channel. The models are classical machine-learning baselines; no
neural networks are used.

## Setup

Use Python 3.10+ and install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Canonical data contract

The model expects one row per time sample, with these columns:

| Column | Required | Meaning |
|---|---|---|
| `dataset` | Yes | Dataset/source identifier |
| `asset_id` | Yes | Bearing, motor, gearbox, or run-to-failure unit |
| `run_id` | Yes | Acquisition run / operating condition |
| `timestamp` | Yes | Monotonic sample time (or sample index) |
| `fault_type` | Yes | Label such as `normal`, `bearing_fault`, or `gear_fault` |
| `rul_seconds` | No | Remaining life in seconds; blank for classification-only data |
| sensor columns | Yes | Numeric channels, e.g. `vibration_x`, `current_a`, `temperature` |

`prepare` converts flat CSV files into this schema. Each source's native layout
is different (including MAT files and multi-file run-to-failure traces), so
export/reshape those files into CSV samples first. Set labels and asset/run IDs
correctly; do not combine unrelated machines under one asset ID. The tool does
not invent fault labels. It can derive a relative time-to-end target for a
run-to-failure trace, but that is only a meaningful RUL target when the trace
actually ends at failure and the sample rate is known.

## Example workflow

Prepare a labelled dataset. For directory-based labels, keep one fault class
per subdirectory, e.g. `raw\cwru\normal\...csv` and
`raw\cwru\inner_race\...csv`:

```powershell
python -m fault_detection.cli prepare `
  --input raw\cwru `
  --output data\cwru.csv `
  --dataset cwru `
  --sensor-columns vibration `
  --asset-column asset_id `
  --run-column run_id `
  --time-column timestamp
```

Alternatively, if every input file has the same label:

```powershell
python -m fault_detection.cli prepare `
  --input raw\mfpt\healthy.csv `
  --output data\mfpt_healthy.csv `
  --dataset mfpt `
  --sensor-columns x `
  --fault-label normal `
  --asset-column bearing_id `
  --sample-rate 97656 `
  --derive-rul
```

Prepare each dataset to the same canonical schema, then combine compatible
canonical CSVs (preserving `asset_id`, sensor columns, and `fault_type`):

```powershell
python -m fault_detection.cli combine `
  --inputs data\cwru.csv data\mfpt.csv `
  --output data\combined.csv
```

Datasets that do not contain the same sensor channels in the same order are
rejected instead of silently mixing incompatible measurements. Map/resample
channels deliberately before combining. Keep dataset identity and operating
conditions available for later evaluation.

Train all applicable model heads:

```powershell
python -m fault_detection.cli train `
  --input data\combined.csv `
  --output-dir models\baseline `
  --window-size 1024 `
  --stride 512 `
  --sample-rate 12000
```

Score another canonical CSV prepared with the **same sensor columns/order**:

```powershell
python -m fault_detection.cli predict `
  --input data\new_run.csv `
  --model-dir models\baseline `
  --output reports\predictions.csv
```

The output includes per-window predicted fault/confidence, anomaly flag/score,
and predicted RUL when those model heads could be trained. Training metrics are
written to `models\baseline\metrics.json`.

## Source-specific training

After obtaining the official archives locally, train FEMTO/PRONOSTIA and the
six-class MAFAULDA model with:

```powershell
python -m fault_detection.femto `
  --archive data\raw\femto\femto_bearing.zip

python -m fault_detection.mafaulda `
  --raw-dir data\raw\mafaulda
```

The FEMTO trainer uses its complete training bearings and complete
`Full_Test_Set` trajectories; it does not treat the truncated challenge
`Test_set.zip` trajectories as supervised RUL labels. The MAFAULDA trainer
decimates the six accelerometer channels from 50 kHz to 1 kHz and validates by
holding out exact filename-speed groups. Both commands save a model, metrics,
and held-out predictions under `models\` and `reports\`.

## Important limitations

- Scores apply only to the named source, channels, and split; they do not
  establish performance on field machines or other datasets.
- Classification needs at least two fault classes. Anomaly detection needs
  healthy/normal windows. RUL needs `rul_seconds` (or verified run-to-failure
  data with `--derive-rul` and the correct sampling rate).
- RUL is only as reliable as the target definition; the example derived target
  is time to the last sample, not automatically time to failure.
- Validation splits are grouped by asset to reduce leakage. With a single
  asset, the model can still train, but no independent holdout metric is
  reported.
- This is a research baseline, not a certified safety or protection system.
  Validate on held-out machines, operating loads, and sensor installations
  before using it to make maintenance decisions.
- Raw downloads and canonical datasets remain local and are not committed.
  Published model artifacts and their metrics are stored separately.
