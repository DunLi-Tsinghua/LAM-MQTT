from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(script: str) -> None:
    cmd = [sys.executable, str(PROJECT_ROOT / "code" / script)]
    print(f"[LAM-MQTT] running {' '.join(cmd)}")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    pipeline = [
        "inspect_datasets.py",
        "preprocess.py",
        "aggregate_flow_features.py",
        "build_feature_views.py",
        "audit_leakage.py",
        "train_baselines.py",
        "run_stage2B_experiments.py",
    ]
    for script in pipeline:
        run(script)
    print("[LAM-MQTT] reproduction pipeline completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
