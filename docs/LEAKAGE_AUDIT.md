# Leakage Audit

LAM-MQTT separates hard leakage from legitimate encrypted-flow metadata before model training.

Hard leakage fields are removed because they encode labels, capture identity, scenario identity, raw endpoint identity, exact time, credentials, or content that is unavailable under encrypted or privacy-constrained monitoring.

Legitimate metadata is retained when it describes observable behavior, such as timing, byte counts, packet counts, rates, burstiness, and TCP flags. A feature may be highly predictive because an attack changes behavior; high single-feature predictiveness alone is not a reason to mark it as leakage.

The leakage audit outputs are stored under:

```text
results/tables/leakage_audit_report_revised.csv
results/tables/hard_leakage_exclusion_rules.csv
results/tables/highly_discriminative_metadata_report.csv
```

These files record audit decisions and sensitivity-analysis candidates.
