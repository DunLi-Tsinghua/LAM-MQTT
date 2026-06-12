from __future__ import annotations

from run_stage2B_experiments import main as run_paper_level_experiments


def main() -> int:
    print(
        "Cross-dataset metrics are produced by the paper-level experiment driver. "
        "Running code/run_stage2B_experiments.py."
    )
    return run_paper_level_experiments()


if __name__ == "__main__":
    raise SystemExit(main())
