# Reproducibility Report

Stage: 2 complete experiment expansion.

No manuscript text, abstract, introduction, or conclusion was generated.

## Commands

```powershell
python code/run_phase2_experiments.py --seed 42
```

## Key Outputs

- `results/metrics/MQTTEEB-D_within_dataset_full_results.csv`
- `results/metrics/MQTTEEB-D_multiclass_results.csv`
- `results/metrics/IoT-23_within_dataset_binary_results.csv`
- `results/metrics/external_validation_MQTTEEB-D_to_IoT-23.csv`
- `results/metrics/multiseed_stability_results.csv`
- `results/tables/feature_importance_top15.csv`
- `results/results_summary.md`
- `results/experiment_log.md`

## Notes

- IoT-23 is treated only as external IoT validation / robustness stress test.
- External validation is not described as MQTT-to-MQTT generalization.
- Raw payload, raw topic, username, password, client ID, exact timestamp, source file, labels, and attack-category columns were not used as model features.

## Local MQTT-IoT-IDS2020 Update

The locally supplied `biflow_features.zip` and `uniflow_features.zip` were processed without any download. `biflow_features.zip` was selected as the primary flow-level source; `uniflow_features.zip` was kept raw and not used for the main rerun.

New/updated outputs:

- `data/processed/MQTT-IoT-IDS2020__cleaned.csv`
- `results/tables/MQTT-IoT-IDS2020_selected_biflow_profile.csv`
- `results/tables/MQTT-IoT-IDS2020_selected_biflow_label_distribution.csv`
- `results/metrics/within_dataset_baseline_results.csv`
- `results/metrics/external_dataset_baseline_results.csv`

MQTT-IoT-IDS2020 biflow cleaned rows: 259,379. Binary distribution: benign 188,378; attack 71,001.


## Stage 2A Commands

```powershell
python code/build_feature_views.py --seed 42
python code/audit_leakage.py --seed 42
python code/build_feature_views.py --seed 42
python code/split_strategies.py --seed 42
python code/run_stage2A_experiments.py --seed 42
```


## Stage 2B Commands

```powershell
python code/build_feature_views.py --seed 42
python code/audit_leakage.py --seed 42
python code/build_feature_views.py --seed 42
python code/run_stage2B_experiments.py --seed 42
```
