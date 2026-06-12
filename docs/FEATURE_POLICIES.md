# Feature Policies

LAM-MQTT evaluates metadata-only intrusion detection under explicit feature policies.

## audited_metadata_main

The main audited policy removes hard leakage and sensitive raw fields while retaining legitimate encrypted-flow metadata.

## conservative_anti_leakage

The conservative policy removes additional highly discriminative metadata as a sensitivity analysis. This is not the main deployment claim.

## common_cross_dataset_metadata

The common policy aligns semantically equivalent metadata features across MQTT datasets for cross-dataset transfer.

## Hard Leakage Removed

```text
label
attack category
scenario
source_file
raw IP identity
exact timestamp
client ID
username/password
payload/topic raw strings
```

## Legitimate Metadata Retained

```text
duration
packet count
byte statistics
packet-size statistics
IAT statistics
rates
burstiness
TCP flags
```

High single-feature predictiveness alone is not treated as leakage.
