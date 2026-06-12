# Label Mapping Rules

## Binary Mapping

- Benign, normal, legitimate, and equivalent labels map to `0`.
- Attack, malicious, anomalous, and attack-family labels map to `1`.

## Multiclass Mapping

- Dataset-provided attack names or detailed labels are retained when available.
- For MQTTEEB-D flow-window aggregation, the binary window label is attack if any packet in the window is attack.
- For MQTTEEB-D flow-window aggregation, the multiclass window label is the majority label in the window.

Label and attack-name fields are used only to construct targets. They are not used as model input features.
