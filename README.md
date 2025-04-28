# simonaMarkovLoad

**Synthetic household load profiles for Simona**

---

## 📖 Description

`simonaMarkovLoad` generates realistic, synthetic household load curves for the [Simona](https://github.com/ie3-institute/simona) simulation environment. It uses time-inhomogeneous Markov chains to model *when* state transitions occur and Gaussian Mixture Models to determine *how much* load is drawn within each state.
## ⚙️ Features

- Discretization of 15-minute consumption values into 10 states
- Bucket concept (month, weekend flag, time of day) for time-dependent transition matrices
- Laplace-smoothed Markov transition matrix per bucket
- GMM fitting per bucket and state with automatic component selection via BIC
- Parallel training with `joblib`
- CLI interface built with Typer/Hydra
- Export synthetic profiles as CSV or integrate directly into Simona API

## 🚀 Installation
### Prerequisites
- **Python** ≥ 3.10
- **Poetry** installed
  - `pip install poetry`  
  - or, preferably, via [pipx](https://pipxproject.github.io/):
   `pipx install poetry`

### Clone the repository
```bash
git clone https://github.com/ie3-institute/simonaMarkovLoad
cd simonamarkovload
```

### Install dependencies & enable hooks
```bash
# Install all dependencies (including dev-dependencies)
poetry install

# Install and activate the pre-commit hooks
poetry run setup
```

## ▶️ Usage

## 📄 License

---

## 📬 Questions?

If you have any questions, feel free to reach out to:

**Philipp Schmelter**  
✉️ <philipp.schmelter@tu-dortmund.de>
---
