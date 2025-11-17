# Repository Guidelines

## Project Structure & Module Organization
Core Python lives in `src/` (`data/generators`, `models/{anomaly_detection,cascading,explainable}`, `analysis`, `alerts`, `visualization` for Streamlit widgets). Baseline settings reside in `config/` (start with `config/config.yaml`). Versioned datasets stay in `data/` (`raw/`, `processed/`, `models/`) with larger fixtures under `TestData/`. Tooling scripts sit in `scripts/`, research assets in `notebooks/`, and regression checks in the root `test_*.py` files plus the scaffolded `tests/` package.

## Build, Test, and Development Commands
- `python install_deps.py` installs the dependency tiers and flags optional failures.
- `python run_dashboard.py` serves the Streamlit app at `http://localhost:8501`.
- `python main.py --mode analysis --output results` runs the full pipeline and writes artifacts to `results/`.
- `python main.py --mode test` performs the lightweight system smoke test.
- `python test_basic.py`, `python test_structure.py`, and peers validate configs/imports; run at least one before a PR.
- `python create_test_data.py --output data/processed` regenerates sample datasets whenever scenarios change.

## Coding Style & Naming Conventions
Target Python 3.11 (3.9+ works) and stick to PEP 8 with four-space indentation. Use `snake_case` for modules and functions, `CamelCase` for classes, and keep YAML keys aligned with `config/config.yaml` (e.g., `network.node_count`). Prefer typed functions and dataclasses (see `src/data/generators/network_generator.py`), encapsulate Streamlit helpers under `src/visualization`, and fence side effects with `if __name__ == "__main__":` guards.

## Testing Guidelines
Name new checks `test_<feature>.py` and keep them runnable via `python test_<name>.py`. Reuse generators from `src/data/generators/` or assets in `TestData/` instead of hitting live systems, and seed randomness so metrics stay deterministic. Document dashboard scenarios by echoing the command or config snippet exercised.

## Commit & Pull Request Guidelines
Recent history uses concise, action-oriented subjects (`AI大模型集成 V0.1`, `first version`); keep titles under ~60 characters and expand details in short bullet bodies when needed. PRs should link issues, highlight schema/data updates, attach UI screenshots when Streamlit changes, and paste output from the relevant test or `main.py` run.

## Configuration & Data Tips
Load secrets from environment variables or a local `.env` (supported via `python-dotenv`); never commit credentials. Update `config/config.yaml` deliberately and note default changes in PR descriptions. Store bulky artifacts in `TestData.zip` and record reproduction commands beside any new `data/processed` exports.
