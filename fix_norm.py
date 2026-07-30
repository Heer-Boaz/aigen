import re

with open('aigen/pix2pix/model.py', 'r') as f:
    content = f.read()

# Replace _batch_norm implementation
content = content.replace(
"""def _batch_norm(
    channels: int,
    *,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
) -> nn.Module:
    return nn.BatchNorm2d(
        channels,
        affine=True,
        track_running_stats=True,
        device=device,
        dtype=dtype,
    )""",
"""def _batch_norm(
    channels: int,
    *,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
) -> nn.Module:
    # Literature fix: Pix2Pix at batch_size=1 fails on validation if BatchNorm uses global running stats.
    # Using InstanceNorm2d (or BatchNorm2d with track_running_stats=False) is required.
    return nn.InstanceNorm2d(
        channels,
        affine=False,
        track_running_stats=False,
        device=device,
        dtype=dtype,
    )"""
)

# Also fix the discriminator BatchNorms
content = re.sub(
    r'nn\.BatchNorm2d\(\n\s+channels \* multiplier,\n\s+device=device,\n\s+dtype=dtype,\n\s+\)',
    r'nn.InstanceNorm2d(\n                        channels * multiplier,\n                        affine=False,\n                        track_running_stats=False,\n                        device=device,\n                        dtype=dtype,\n                    )',
    content
)

# Fix weight initialization for InstanceNorm2d (which has no weights if affine=False)
content = content.replace(
"""    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, mean=1.0, std=0.02)
        nn.init.zeros_(module.bias)""",
"""    elif isinstance(module, nn.BatchNorm2d):
        if module.weight is not None:
            nn.init.normal_(module.weight, mean=1.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.InstanceNorm2d):
        if module.weight is not None:
            nn.init.normal_(module.weight, mean=1.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)"""
)

with open('aigen/pix2pix/model.py', 'w') as f:
    f.write(content)

print("model.py patched successfully")
