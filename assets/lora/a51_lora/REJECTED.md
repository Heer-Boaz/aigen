# Rejected Dataset

This package is forensic evidence only.

The selected images are not canon-worthy enough for AI51 identity LoRA training:
identity, outfit, proportions and image quality are inconsistent across the set.

Do not use this dataset or the trained weights as evidence against LoRA quality.
Use a canon-first dataset instead:

```bash
.venv/bin/python -m aigen.cli lora canon-init ...
.venv/bin/python -m aigen.cli lora dataset-audit ...
```
