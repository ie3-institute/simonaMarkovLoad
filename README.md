# simonaMarkovLoad

**Synthetic household load profiles for Simona**

---

## 📖 Description

`simonaMarkovLoad` generates realistic, synthetic household load curves for the [Simona](https://github.com/ie3-institute/simona) simulation environment. It uses **time-inhomogeneous Markov chains** to model *when* state transitions occur and **Gaussian Mixture Models (GMMs)** to determine *how much* load is drawn within each state.

The trained model is exported in the **PSDM JSON format** (`simonaMarkovLoad:psdm:1.0`) for direct integration with Simona.

---

## ⚙️ Features

- Conversion of cumulative 15-minute kWh readings to instantaneous power (kW) via differencing
- Discretisation of continuous power values into **10 states** using quadratic thresholds
- **2,304 temporal buckets** based on month, weekday/weekend flag, and quarter-hour slot
- **Laplace-smoothed Markov transition matrices** per bucket (handles sparse data gracefully)
- **GMM fitting** per (bucket, state) pair with automatic component selection via BIC (1–3 components)
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
| `Messwert` | Cumulative energy reading in kWh (configurable) |

The first 21 rows are skipped by default (configurable via `config.yml`).

### 2. Configure

Edit `config.yml` to match your data format:

```yaml
input:
  skiprows: 21              # Header rows to skip
  timestamp_col: "Zeitstempel"
  value_col: "Messwert"
  factor: 4                 # Conversion factor (15-min intervals → kW)

model:
  n_states: 10              # Number of load states
  laplace_alpha: 1.0        # Laplace smoothing for transition matrices
```

### 3. Run the pipeline

```bash
poetry run python -m src.main
```

This will:

1. Load and preprocess raw CSV data
2. Assign temporal buckets to each observation
3. Build Markov transition matrices (2,304 × 10 × 10)
4. Fit GMMs for each (bucket, state) pair in parallel
5. Export the model to `out/psdm_model.json`

### 4. Output

The model is written to `out/psdm_model.json` by default.

```
{
  "schema": "simonaMarkovLoad:psdm:1.0",
  "meta": { ... },
  "timeModel": {
    "nBuckets": 2304,
    "bucketFormula": "month*192 + is_weekend*96 + quarter_hour"
  },
  "valueModel": { "nStates": 10, "stateThresholds": [ ... ] },
  "data": {
    "transitionMatrices": [ ... ],
    "gmms": [ ... ]
  }
}
```

---

## 🔍 How it works

### Bucketing

Each 15-minute observation is assigned to one of **2,304 buckets**:

```
bucket = month * 192 + is_weekend * 96 + quarter_hour
```

This allows transition probabilities and load distributions to vary by time of day, day type (weekday vs. weekend), and season.

### Markov chain

For each bucket a 10×10 transition matrix is estimated from historical data. Rows with no observed transitions default to a **self-loop** (stay in the current state), ensuring valid probability distributions in all cases.

### GMM sampling

Within each (bucket, state) pair a GMM models the distribution of normalised power values. The number of components (1–3) is selected automatically by minimising BIC. At inference time a component is chosen according to its mixture weight, then a value is drawn from the corresponding Gaussian and clamped to [0, 1].

---

## 🛠️ Development

### Running tests

```bash
poetry run pytest
```

### Code quality

Pre-commit hooks enforce:

- **black** – code formatting (line length 88)
- **isort** – import ordering
- **ruff** – linting (E, F, W, I, N, UP, B, C4, PIE, SIM, RET rules)

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| numpy | >=2.2.5,<3 | Numerical arrays |
| pandas | >=2.2.3,<3 | Data manipulation |
| scikit-learn | >=1.6.1,<2 | Gaussian Mixture Models |
| joblib | >=1.4.2,<2 | Parallel GMM fitting |
| matplotlib | >=3.10.1,<4 | Visualisation |
| tqdm | >=4.67.1,<5 | Progress bars |

---

## 📄 License

---

## 📬 Questions?

If you have any questions, feel free to reach out to:

**Philipp Schmelter** — philipp.schmelter@tu-dortmund.de
