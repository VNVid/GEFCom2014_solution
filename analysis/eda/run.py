"""Command-line entry point for the reproducible load-track EDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .analysis import analyze
from .figures import generate_figures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/eda.yaml"))
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {args.config}")

    load_dir = Path(config["data"]["load_dir"])
    figures_dir = Path(config["outputs"]["figures_dir"])
    tables_dir = Path(config["outputs"]["tables_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    results = analyze(load_dir, config)
    for filename, table in results.tables.items():
        table.to_csv(tables_dir / filename, index=False, float_format="%.6f")

    with (tables_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(results.summary, stream, indent=2, sort_keys=True)
        stream.write("\n")

    generate_figures(results, figures_dir, config["plot"])
    print(json.dumps(results.summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
