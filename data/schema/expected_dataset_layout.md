# Expected Dataset Layout

```text
data/
  raw/
    MQTTEEB-D/
      <downloaded MQTTEEB-D CSV files or extracted official folder>
    MQTT-IoT-IDS2020/
      biflow_features.zip
      uniflow_features.zip
    IoT-23/
      <labelled Zeek flow logs from Stratosphere Laboratory>
  interim/
    <created by scripts>
  processed/
    <created by scripts>
```

The repository does not redistribute raw archives, packet captures, Zeek logs, or processed parquet files.
