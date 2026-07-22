# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `CODEOWNERS` file and dependabot automation [#7](https://github.com/ie3-institute/simonaMarkovLoad/issues/7)
- Added full Markov-model pipeline [#10](https://github.com/ie3-institute/simonaMarkovLoad/issues/10)
- Added GMM feature to project [#12](https://github.com/ie3-institute/simonaMarkovLoad/issues/12)
- Added `JSON` export and improve simulation robustness [#27](https://github.com/ie3-institute/simonaMarkovLoad/issues/27)
- Per-file constant-load triples stored in `constant_loads.yml`, mapped by CSV file stem [#125](https://github.com/ie3-institute/simonaMarkovLoad/issues/125)
- Optional pooled training: `input.pools: true` trains one model per `data/` subdirectory and exports one PSDM JSON per pool [#127](https://github.com/ie3-institute/simonaMarkovLoad/issues/127)
- `dropped_missing_values` metadata counter on the loader output

### Fixed
- Transition counts are now computed per source file and aggregated, preventing spurious cross-household transitions [#83](https://github.com/ie3-institute/simonaMarkovLoad/issues/83)
- `skiprows` is now read from config instead of being hardcoded [#83](https://github.com/ie3-institute/simonaMarkovLoad/issues/83)
- Removed duplicate `_core.py`; `build_transition_matrices` accepts pre-computed counts to avoid redundant computation [#83](https://github.com/ie3-institute/simonaMarkovLoad/issues/83)
- Tracked sample file `dummy.csv` is no longer silently included in the training data [#127](https://github.com/ie3-institute/simonaMarkovLoad/issues/127)

### Changed
- Compute instantaneous kW from cumulative kWh via 15-minute differencing [#1](https://github.com/ie3-institute/simonaMarkovLoad/issues/1)
- Replaced `input.factor` with `input.value_representation` (`cumulative_energy`, `interval_energy`, `power`) and `input.interval_minutes`; the legacy key is rejected with a clear error
- Moved the data root from `data/raw/` to `data/` and removed `data/processed/`; single-model mode accepts loose CSVs or exactly one subdirectory [#127](https://github.com/ie3-institute/simonaMarkovLoad/issues/127)
- Migrated dependency management, packaging, and CI from Poetry to uv [#129](https://github.com/ie3-institute/simonaMarkovLoad/issues/129)
