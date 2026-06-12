# LAM-MQTT

## Paper

**Leakage-Audited Metadata-Only Intrusion Detection for Encrypted MQTT-Based IoT Traffic**

This repository provides the code and reproducibility materials for LAM-MQTT. LAM-MQTT evaluates leakage-audited metadata-only intrusion detection for encrypted MQTT-based IoT traffic. The code does not decrypt TLS and does not use payloads, raw topics, credentials, raw client identifiers, raw IP identities, or exact timestamps as model inputs.

## Datasets

Raw datasets are not redistributed in this repository. Obtain them from the official sources and place them under `data/raw/`.

- MQTTEEB-D: Mendeley Data, https://data.mendeley.com/datasets/jfttfjn6tr/1
- MQTT-IoT-IDS2020: official dataset page, https://pureportal.strath.ac.uk/en/datasets/mqtt-iot-ids2020-mqtt-internet-of-things-intrusion-detection-data/
- IoT-23: Stratosphere Laboratory, https://www.stratosphereips.org/datasets-iot23

Expected raw-data layout:

```text
data/raw/MQTTEEB-D/
data/raw/MQTT-IoT-IDS2020/
data/raw/IoT-23/
```

For MQTT-IoT-IDS2020, place `biflow_features.zip` in `data/raw/MQTT-IoT-IDS2020/`. The optional `uniflow_features.zip` can be placed in the same folder.

## Quick Reproduction

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

Run the main pipeline after placing raw datasets:

```bash
python code/inspect_datasets.py
python code/preprocess.py
python code/aggregate_flow_features.py
python code/build_feature_views.py
python code/audit_leakage.py
python code/train_baselines.py
python code/cross_dataset_eval.py
python code/feature_family_ablation.py
python code/plot_results.py
```

One-command reproduction:

```bash
python code/run_all.py
```

Windows:

```bat
scripts\reproduce_stage2B.bat
```

## Repository Contents

- `code/`: dataset inspection, preprocessing, leakage audit, feature-view construction, baseline training, cross-dataset evaluation wrappers, and figure/result scripts.
- `configs/`: default paths and feature policy definitions.
- `data/schema/`: expected dataset layout and metadata feature schema.
- `results/`: lightweight metric tables, analysis tables, and manuscript figures.
- `paper/`: submitted manuscript source and preprint PDF snapshot.
- `docs/`: reproducibility, dataset, feature-policy, and leakage-audit notes.

## Citation

```bibtex
@misc{lam_mqtt_2026,
  title = {LAM-MQTT: Leakage-Audited Metadata-Only Intrusion Detection for Encrypted MQTT-Based IoT Traffic},
  author = {Li, Dun and Li, Hongzhi and Crespi, Noel and Minerva, Roberto and Li, Ming and Liang, Wei and Li, Kuan-Ching},
  year = {2026},
  howpublished = {\url{https://github.com/DunLi-Tsinghua/LAM-MQTT}}
}
```

## License

Code and documentation are released under the MIT License. Dataset redistribution is governed by the original dataset providers.
