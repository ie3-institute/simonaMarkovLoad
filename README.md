# simonaMarkovLoad

**Synthetic household load profiles for Simona**

---

## 📖 Description

`simonaMarkovLoad` generates realistic, synthetic household load curves for the [Simona](https://github.com/ie3-institute/simona) simulation environment. It uses **time-inhomogeneous Markov chains** to model *when* state transitions occur and **Gaussian Mixture Models (GMMs)** to determine *how much* load is drawn within each state.

The trained model is exported in the **PSDM JSON format** (`simonaMarkovLoad:psdm:1.0`) for direct integration with Simona.

---

## ⚙️ Features

- Semantic handling of cumulative energy, interval energy, and power input values
- **Global min/max normalisation** across all input files (`minmax_global`); the kW scale is exported with the model so consumers can de-normalise
- Discretisation of continuous power values into **10 states** using quadratic thresholds
- **2,304 temporal buckets** based on month, weekday/weekend flag, and quarter-hour slot
- **Markov transition matrices** per bucket with a **self-loop fallback** for empty rows (handles sparse data gracefully)
- **GMM fitting** per (bucket, state) pair with automatic component selection via BIC (1-3 components)
- Parallel training with `joblib`
- Export of the complete model as a **PSDM JSON** file
- Built-in simulation and diagnostic visualisations

---

## 🗂️ Project Structure

```
simonaMarkovLoad/
├── src/
│   ├── config.py               # YAML configuration loader
│   ├── main.py                 # Full pipeline: load → train → export → simulate
│   ├── export.py               # PSDM JSON export
│   ├── markov/
│   │   ├── buckets.py          # Temporal bucket assignment (2,304 buckets)
│   │   ├── gmm.py              # GMM fitting and sampling
│   │   ├── transition_counts.py # Raw transition count matrices
│   │   └── transitions.py      # Probability matrices with self-loop fallback
│   └── preprocessing/
│       ├── loader.py           # CSV ingestion and power computation
│       └── scaling.py          # Normalisation and discretisation
├── tests/                      # pytest test suite (9 modules)
├── scripts/
│   └── setup_env.py            # Pre-commit hook installer
├── data/
│   ├── raw/                    # Input CSV files (place your data here)
│   └── processed/
├── out/                        # Model output (psdm_model.json)
├── config.yml                  # Configuration
└── pyproject.toml
```

---

## 🚀 Installation

### Prerequisites

- **Python** >= 3.13
- **Poetry** installed:
  ```bash
  pip install poetry
  # or, preferably, via pipx:
  pipx install poetry
  ```

### Clone and install

```bash
git clone https://github.com/ie3-institute/simonaMarkovLoad
cd simonaMarkovLoad

# Install all dependencies (including dev dependencies)
poetry install

# Install and activate pre-commit hooks
poetry run setup
```

To run the hooks manually:

```bash
poetry run pre-commit run --all-files
```

---

## ▶️ Usage

### 1. Prepare input data

Place your raw CSV files in `data/raw/`. The CSV format is expected to have:

| Column | Description |
|---|---|
| `Zeitstempel` | Timestamp (configurable) |
| `Messwert` | Energy in kWh or power in kW (configurable) |

The first 21 rows are skipped by default (configurable via `config.yml`).

### 2. Configure

Edit `config.yml` to match your data format:

```yaml
input:
  skiprows: 21              # Header rows to skip
  timestamp_col: "Zeitstempel"
  value_col: "Messwert"
  value_representation: "cumulative_energy"
  interval_minutes: 15      # Required for energy input
  drop_negative_deltas: true # Drop invalid meter resets/corrections

model:
  n_states: 10              # Number of load states

gmm:
  n_jobs: -1                # Parallel workers for GMM fitting
  random_state: 42          # Reproducible GMM initialisation

output:
  psdm_json: "out/psdm_model.json"
  show_plots: true          # Set false for headless training runs
```

`value_representation` defines how the configured value column is interpreted:

| Value | Input | Conversion to kW |
|---|---|---|
| `cumulative_energy` | Cumulative energy meter reading in kWh | Difference consecutive readings, then divide by the interval duration in hours |
| `interval_energy` | Energy consumed during each interval in kWh | Divide each value by the interval duration in hours |
| `power` | Power in kW | Use each value directly |

`interval_minutes` is required for both energy representations and must be
positive. In `power` mode it is not used in the calculation, but when specified
it is still validated as a positive, finite value. The temporal model currently
assumes fixed 15-minute intervals, so `interval_minutes` must remain `15`! it
controls only the energy to power scaling. `drop_negative_deltas` is likewise
allowed in every mode but is applied only to `cumulative_energy`. When omitted
for that mode, negative deltas are dropped by default.

#### Per-file constant loads

Optional constant-load triples can be stored by CSV file stem in
`constant_loads.yml` at the repository root:

```yaml
SM_00001:
  - [330.0, 20, 6]
SM_00002:
  - [150, 24, 8]
  - [80.0, 12, 2]
```

The values are stored for consumption by future preprocessing steps. Their
interpretation is TBD; see `multimeter` `base_variation`.

### 3. Run the pipeline

```bash
poetry run python -m src.main
```

This will:

Negative cumulative meter deltas are dropped by default before normalisation, because they usually indicate meter resets or data corrections.

1. Load and preprocess raw CSV data (input values → kW, global min/max normalisation, discretisation)
2. Assign temporal buckets to each observation
3. Build Markov transition matrices (2,304 × 10 × 10)
4. Fit GMMs for each (bucket, state) pair in parallel
5. Export the model to `out/psdm_model.json`
6. Run a short simulation and display diagnostic plots

### 4. Output

The model is written to `out/psdm_model.json` by default.

```
{
  "schema": "simonaMarkovLoad:psdm:1.0",
  "generated_at": "...",
  "generator": { "name": "simonaMarkovLoad", "config": { ... } },
  "time_model": {
    "bucket_count": 2304,
    "bucket_encoding": { "formula": "bucket = month*192 + is_weekend*96 + quarter_hour" },
    "sampling_interval_minutes": 15,
    "timezone": "Europe/Berlin"
  },
  "value_model": {
    "value_unit": "normalized",
    "normalization": {
      "method": "minmax_global",
      "max_power": { "value": ..., "unit": "kW" },
      "min_power": { "value": ..., "unit": "kW" }
    },
    "discretization": { "states": 10, "thresholds_right": [ ... ] }
  },
  "parameters": {
    "transitions": { "empty_row_strategy": "self_loop" },
    "gmm": { ... }
  },
  "data": {
    "transitions": { "shape": [2304, 10, 10], "dtype": "float32", "values": [ ... ] },
    "gmms": { "buckets": [ { "states": [ { "weights": [...], "means": [...], "variances": [...] }, ... ] }, ... ] }
  },
  "training_data": { "records": ..., "time_range": { ... } }
}
```

---

## 🔍 How it works

### Normalisation

Instantaneous power values from **all** input files are normalised together using the global minimum and maximum (`minmax_global`). The kW scale (`min_power`, `max_power`) is written to the exported JSON so that consumers such as Simona can map normalised values back to physical power.

### Bucketing

Each 15-minute observation is assigned to one of **2,304 buckets**:

```
bucket = month * 192 + is_weekend * 96 + quarter_hour
```

This allows transition probabilities and load distributions to vary by time of day, day type (weekday vs. weekend), and season.

### Markov chain

For each bucket a 10×10 transition matrix is estimated from historical data. Rows with no observed transitions default to a **self-loop** (stay in the current state), ensuring valid probability distributions in all cases.

### GMM sampling

Within each (bucket, state) pair a GMM models the distribution of normalised power values. The number of components (1-3) is selected automatically by minimising BIC. At inference time a component is chosen according to its mixture weight, then a value is drawn from the corresponding Gaussian and clamped to [0, 1].

---

## 🛠️ Development

### Running tests

```bash
poetry run pytest
```

### Code quality

Pre-commit hooks enforce:

- **black** - code formatting (line length 88)
- **isort** - import ordering
- **ruff** - linting (E, F, W, I, N, UP, B, C4, PIE, SIM, RET rules)

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| numpy | >=2.5.0,<3 | Numerical arrays |
| pandas | >=2.2.3,<3 | Data manipulation |
| scikit-learn | >=1.9.0,<2 | Gaussian Mixture Models |
| joblib | >=1.4.2,<2 | Parallel GMM fitting |
| matplotlib | >=3.11.0,<4 | Visualisation |
| tqdm | >=4.68.3,<5 | Progress bars |

---

## 📄 License

---

## 📬 Questions?

If you have any questions, feel free to reach out to:

**Philipp Schmelter** - philipp.schmelter@tu-dortmund.de
