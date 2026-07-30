from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from aigen.pix2pix.config import ModelConfig


NormalizationFactory = Callable[[int], nn.Module]


class Pix2PixGenerator(nn.Module):
    """Reference pix2pix U-Net generator for a power-of-two square canvas."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        norm = lambda channels: _batch_norm(
            channels,
            device=device,
            dtype=dtype,
        )
        channels = config.generator_channels
        block = _UnetBlock(
            channels * 8,
            channels * 8,
            innermost=True,
            norm=norm,
            device=device,
            dtype=dtype,
        )
        num_downs = config.image_size.bit_length() - 1
        for _ in range(num_downs - 5):
            block = _UnetBlock(
                channels * 8,
                channels * 8,
                child=block,
                norm=norm,
                dropout=config.generator_dropout,
                device=device,
                dtype=dtype,
            )
        block = _UnetBlock(
            channels * 4,
            channels * 8,
            child=block,
            norm=norm,
            device=device,
            dtype=dtype,
        )
        block = _UnetBlock(
            channels * 2,
            channels * 4,
            child=block,
            norm=norm,
            device=device,
            dtype=dtype,
        )
        block = _UnetBlock(
            channels,
            channels * 2,
            child=block,
            norm=norm,
            device=device,
            dtype=dtype,
        )
        self.network = _UnetBlock(
            config.output_channels,
            channels,
            input_channels=config.input_channels,
            child=block,
            outermost=True,
            norm=norm,
            device=device,
            dtype=dtype,
        )

    def forward(self, source: Tensor) -> Tensor:
        return self.network(source)


class ConditionalPatchDiscriminator(nn.Module):
    """Reference conditional N-layer PatchGAN discriminator."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        input_channels = config.input_channels + config.output_channels
        channels = config.discriminator_channels
        layers: list[nn.Module] = [
            nn.Conv2d(
                input_channels,
                channels,
                kernel_size=4,
                stride=2,
                padding=1,
                device=device,
                dtype=dtype,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        feature_indices = [len(layers) - 1]
        multiplier = 1
        for layer_index in range(1, config.discriminator_layers):
            previous = multiplier
            multiplier = min(2**layer_index, 8)
            layers.extend(
                [
                    nn.Conv2d(
                        channels * previous,
                        channels * multiplier,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                        device=device,
                        dtype=dtype,
                    ),
                    nn.InstanceNorm2d(
                        channels * multiplier,
                        affine=False,
                        track_running_stats=False,
                        device=device,
                        dtype=dtype,
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            feature_indices.append(len(layers) - 1)
        previous = multiplier
        multiplier = min(2**config.discriminator_layers, 8)
        layers.extend(
            [
                nn.Conv2d(
                    channels * previous,
                    channels * multiplier,
                    kernel_size=4,
                    stride=1,
                    padding=1,
                    bias=False,
                    device=device,
                    dtype=dtype,
                ),
                nn.InstanceNorm2d(
                        channels * multiplier,
                        affine=False,
                        track_running_stats=False,
                        device=device,
                        dtype=dtype,
                    ),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(
                    channels * multiplier,
                    1,
                    kernel_size=4,
                    stride=1,
                    padding=1,
                    device=device,
                    dtype=dtype,
                ),
            ]
        )
        self.network = nn.Sequential(*layers)
        feature_indices.append(len(layers) - 2)
        self._feature_indices = tuple(feature_indices)

    def forward(self, source: Tensor, candidate: Tensor) -> Tensor:
        return self.network(torch.cat((source, candidate), dim=1))

    def forward_features(
        self,
        source: Tensor,
        candidate: Tensor,
    ) -> tuple[Tensor, ...]:
        output = torch.cat((source, candidate), dim=1)
        features = []
        next_feature = 0
        for layer_index, layer in enumerate(self.network):
            output = layer(output)
            if (
                next_feature < len(self._feature_indices)
                and layer_index == self._feature_indices[next_feature]
            ):
                features.append(output)
                next_feature += 1
        features.append(output)
        return tuple(features)


class ConditionalMultiscaleDiscriminator(nn.Module):
    """Conditional PatchGANs over native and average-pooled image scales."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        scales: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(
            ConditionalPatchDiscriminator(
                config,
                device=device,
                dtype=dtype,
            )
            for _ in range(scales)
        )
        self.downsample = nn.AvgPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
            count_include_pad=False,
        )

    def forward(
        self,
        source: Tensor,
        candidate: Tensor,
    ) -> tuple[Tensor, ...]:
        logits = []
        for scale_index, discriminator in enumerate(self.discriminators):
            logits.append(discriminator(source, candidate))
            if scale_index + 1 < len(self.discriminators):
                source = self.downsample(source)
                candidate = self.downsample(candidate)
        return tuple(logits)

    def forward_features(
        self,
        source: Tensor,
        candidate: Tensor,
    ) -> tuple[tuple[Tensor, ...], ...]:
        outputs = []
        for scale_index, discriminator in enumerate(self.discriminators):
            outputs.append(discriminator.forward_features(source, candidate))
            if scale_index + 1 < len(self.discriminators):
                source = self.downsample(source)
                candidate = self.downsample(candidate)
        return tuple(outputs)

    def forward_paired(
        self,
        source: Tensor,
        fake: Tensor,
        real: Tensor,
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        batch_size = source.shape[0]
        combined = self.forward(
            torch.cat((source, source), dim=0),
            torch.cat((fake, real), dim=0),
        )
        fake_logits = tuple(logits[:batch_size] for logits in combined)
        real_logits = tuple(logits[batch_size:] for logits in combined)
        return fake_logits, real_logits

    def forward_a_contrario(
        self,
        source: Tensor,
        mismatched_source: Tensor,
        fake: Tensor,
        real: Tensor,
    ) -> tuple[
        tuple[Tensor, ...],
        tuple[Tensor, ...],
        tuple[Tensor, ...],
        tuple[Tensor, ...],
    ]:
        batch_size = source.shape[0]
        combined = self.forward(
            torch.cat(
                (source, mismatched_source, mismatched_source, source),
                dim=0,
            ),
            torch.cat((fake, real, fake, real), dim=0),
        )
        fake_logits = tuple(logits[:batch_size] for logits in combined)
        mismatched_real_logits = tuple(
            logits[batch_size : batch_size * 2] for logits in combined
        )
        mismatched_generated_logits = tuple(
            logits[batch_size * 2 : batch_size * 3] for logits in combined
        )
        real_logits = tuple(logits[batch_size * 3 :] for logits in combined)
        return (
            fake_logits,
            mismatched_real_logits,
            mismatched_generated_logits,
            real_logits,
        )

    def forward_paired_features(
        self,
        source: Tensor,
        fake: Tensor,
        real: Tensor,
    ) -> tuple[
        tuple[tuple[Tensor, ...], ...],
        tuple[tuple[Tensor, ...], ...],
    ]:
        batch_size = source.shape[0]
        combined = self.forward_features(
            torch.cat((source, source), dim=0),
            torch.cat((fake, real), dim=0),
        )
        fake_features = tuple(
            tuple(feature[:batch_size] for feature in scale)
            for scale in combined
        )
        real_features = tuple(
            tuple(feature[batch_size:] for feature in scale)
            for scale in combined
        )
        return fake_features, real_features


class _UnetBlock(nn.Module):
    def __init__(
        self,
        outer_channels: int,
        inner_channels: int,
        *,
        input_channels: int | None = None,
        child: nn.Module | None = None,
        outermost: bool = False,
        innermost: bool = False,
        norm: NormalizationFactory,
        dropout: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.outermost = outermost
        input_channels = input_channels or outer_channels
        down = nn.Conv2d(
            input_channels,
            inner_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
            device=device,
            dtype=dtype,
        )
        if outermost:
            assert child is not None
            self.network = nn.Sequential(
                down,
                child,
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    inner_channels * 2,
                    outer_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    device=device,
                    dtype=dtype,
                ),
                nn.Tanh(),
            )
            return
        if innermost:
            self.network = nn.Sequential(
                nn.LeakyReLU(0.2, inplace=True),
                down,
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    inner_channels,
                    outer_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                    device=device,
                    dtype=dtype,
                ),
                norm(outer_channels),
            )
            return
        assert child is not None
        modules: list[nn.Module] = [
            nn.LeakyReLU(0.2, inplace=True),
            down,
            norm(inner_channels),
            child,
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                inner_channels * 2,
                outer_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
                device=device,
                dtype=dtype,
            ),
            norm(outer_channels),
        ]
        if dropout:
            modules.append(nn.Dropout(0.5))
        self.network = nn.Sequential(*modules)

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.network(inputs)
        if self.outermost:
            return output
        return torch.cat((inputs, output), dim=1)


def initialize_pix2pix_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        if module.weight is not None:
            nn.init.normal_(module.weight, mean=1.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.InstanceNorm2d):
        if module.weight is not None:
            nn.init.normal_(module.weight, mean=1.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def generator_loss(
    fake_logits: Tensor,
    generated: Tensor,
    target: Tensor,
    *,
    lambda_l1: float,
) -> tuple[Tensor, Tensor, Tensor]:
    adversarial = functional.softplus(-fake_logits.float()).mean()
    reconstruction = functional.l1_loss(generated.float(), target.float())
    total = adversarial + reconstruction * lambda_l1
    return total, adversarial, reconstruction


def reconstruction_loss(
    generated: Tensor,
    target: Tensor,
    *,
    lambda_l1: float,
) -> tuple[Tensor, Tensor]:
    reconstruction = functional.l1_loss(generated.float(), target.float())
    return reconstruction * lambda_l1, reconstruction


def discriminator_loss(real_logits: Tensor, fake_logits: Tensor) -> Tensor:
    real = functional.softplus(-real_logits.float()).mean()
    fake = functional.softplus(fake_logits.float()).mean()
    return (real + fake) * 0.5


def multiscale_discriminator_loss(
    real_logits: tuple[Tensor, ...],
    fake_logits: tuple[Tensor, ...],
) -> Tensor:
    return torch.stack(
        tuple(
            discriminator_loss(real_scale, fake_scale)
            for real_scale, fake_scale in zip(real_logits, fake_logits, strict=True)
        )
    ).mean()


def multiscale_generator_loss(
    fake_logits: tuple[Tensor, ...],
    generated: Tensor,
    target: Tensor,
    *,
    lambda_l1: float,
    lambda_feature_matching: float,
    reconstruction_balance: str,
    fake_features: tuple[tuple[Tensor, ...], ...] | None = None,
    real_features: tuple[tuple[Tensor, ...], ...] | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    adversarial = torch.stack(
        tuple(
            functional.softplus(-scale_logits.float()).mean()
            for scale_logits in fake_logits
        )
    ).mean()
    feature_matching = generated.new_zeros((), dtype=torch.float32)
    if lambda_feature_matching > 0:
        assert fake_features is not None and real_features is not None
        per_scale = []
        for fake_scale, real_scale in zip(
            fake_features,
            real_features,
            strict=True,
        ):
            feature_weight = 4.0 / (len(fake_scale) - 1)
            per_scale.append(
                torch.stack(
                    tuple(
                        functional.l1_loss(
                            fake.float(),
                            real.detach().float(),
                        )
                        for fake, real in zip(
                            fake_scale[:-1],
                            real_scale[:-1],
                            strict=True,
                        )
                    )
                ).sum()
                * feature_weight
            )
        feature_matching = torch.stack(tuple(per_scale)).mean()
    reconstruction = functional.l1_loss(generated.float(), target.float())
    reconstruction_for_total = (
        reconstruction
        if reconstruction_balance == "uniform"
        else white_canvas_equal_regions_loss(generated, target)
    )
    total = (
        adversarial
        + feature_matching * lambda_feature_matching
        + reconstruction_for_total * lambda_l1
    )
    return (
        total,
        adversarial,
        reconstruction,
        reconstruction_for_total,
        feature_matching,
    )


@dataclass(frozen=True)
class BalancedDiscriminatorLoss:
    total: Tensor
    real_foreground: Tensor
    real_background: Tensor
    fake_foreground: Tensor
    fake_background: Tensor


@dataclass(frozen=True)
class BalancedGeneratorLoss:
    total: Tensor
    adversarial: Tensor
    adversarial_foreground: Tensor
    adversarial_background: Tensor
    reconstruction: Tensor
    reconstruction_for_total: Tensor
    feature_matching: Tensor
    feature_matching_foreground: Tensor
    feature_matching_background: Tensor


@dataclass(frozen=True)
class AContrarioDiscriminatorLoss:
    total: Tensor
    real_foreground: Tensor
    real_background: Tensor
    fake_foreground: Tensor
    fake_background: Tensor
    mismatched_real_foreground: Tensor
    mismatched_real_background: Tensor
    mismatched_generated_foreground: Tensor
    mismatched_generated_background: Tensor


@dataclass(frozen=True)
class PaletteProximityLoss:
    total: Tensor
    foreground: Tensor
    background: Tensor


def balanced_multiscale_discriminator_loss(
    real_logits: tuple[Tensor, ...],
    fake_logits: tuple[Tensor, ...],
    *,
    foreground_mask: Tensor,
) -> BalancedDiscriminatorLoss:
    real_regions = tuple(
        _white_canvas_region_means(
            functional.softplus(-real_scale.float()),
            foreground_mask,
        )
        for real_scale in real_logits
    )
    fake_regions = tuple(
        _white_canvas_region_means(
            functional.softplus(fake_scale.float()),
            foreground_mask,
        )
        for fake_scale in fake_logits
    )
    real_foreground = torch.stack(
        tuple(region[0] for region in real_regions)
    ).mean()
    real_background = torch.stack(
        tuple(region[1] for region in real_regions)
    ).mean()
    fake_foreground = torch.stack(
        tuple(region[0] for region in fake_regions)
    ).mean()
    fake_background = torch.stack(
        tuple(region[1] for region in fake_regions)
    ).mean()
    total = (
        real_foreground
        + real_background
        + fake_foreground
        + fake_background
    ) * 0.25
    return BalancedDiscriminatorLoss(
        total=total,
        real_foreground=real_foreground,
        real_background=real_background,
        fake_foreground=fake_foreground,
        fake_background=fake_background,
    )


def a_contrario_multiscale_discriminator_loss(
    real_logits: tuple[Tensor, ...],
    fake_logits: tuple[Tensor, ...],
    mismatched_real_logits: tuple[Tensor, ...],
    mismatched_generated_logits: tuple[Tensor, ...],
    *,
    foreground_mask: Tensor,
) -> AContrarioDiscriminatorLoss:
    real_regions = tuple(
        _white_canvas_region_means(
            functional.softplus(-real_scale.float()),
            foreground_mask,
        )
        for real_scale in real_logits
    )
    fake_regions = tuple(
        _white_canvas_region_means(
            functional.softplus(fake_scale.float()),
            foreground_mask,
        )
        for fake_scale in fake_logits
    )
    mismatched_real_regions = tuple(
        _white_canvas_region_means(
            functional.softplus(mismatched_scale.float()),
            foreground_mask,
        )
        for mismatched_scale in mismatched_real_logits
    )
    mismatched_generated_regions = tuple(
        _white_canvas_region_means(
            functional.softplus(mismatched_scale.float()),
            foreground_mask,
        )
        for mismatched_scale in mismatched_generated_logits
    )
    real_foreground = torch.stack(
        tuple(region[0] for region in real_regions)
    ).mean()
    real_background = torch.stack(
        tuple(region[1] for region in real_regions)
    ).mean()
    fake_foreground = torch.stack(
        tuple(region[0] for region in fake_regions)
    ).mean()
    fake_background = torch.stack(
        tuple(region[1] for region in fake_regions)
    ).mean()
    mismatched_real_foreground = torch.stack(
        tuple(region[0] for region in mismatched_real_regions)
    ).mean()
    mismatched_real_background = torch.stack(
        tuple(region[1] for region in mismatched_real_regions)
    ).mean()
    mismatched_generated_foreground = torch.stack(
        tuple(region[0] for region in mismatched_generated_regions)
    ).mean()
    mismatched_generated_background = torch.stack(
        tuple(region[1] for region in mismatched_generated_regions)
    ).mean()
    total = (
        real_foreground
        + real_background
        + fake_foreground
        + fake_background
        + mismatched_real_foreground
        + mismatched_real_background
        + mismatched_generated_foreground
        + mismatched_generated_background
    ) * 0.125
    return AContrarioDiscriminatorLoss(
        total=total,
        real_foreground=real_foreground,
        real_background=real_background,
        fake_foreground=fake_foreground,
        fake_background=fake_background,
        mismatched_real_foreground=mismatched_real_foreground,
        mismatched_real_background=mismatched_real_background,
        mismatched_generated_foreground=mismatched_generated_foreground,
        mismatched_generated_background=mismatched_generated_background,
    )


def balanced_multiscale_generator_loss(
    fake_features: tuple[tuple[Tensor, ...], ...],
    real_features: tuple[tuple[Tensor, ...], ...],
    generated: Tensor,
    target: Tensor,
    *,
    foreground_mask: Tensor,
    lambda_l1: float,
    lambda_feature_matching: float,
    reconstruction_balance: str,
) -> BalancedGeneratorLoss:
    adversarial_regions = tuple(
        _white_canvas_region_means(
            functional.softplus(-scale[-1].float()),
            foreground_mask,
        )
        for scale in fake_features
    )
    adversarial_foreground = torch.stack(
        tuple(region[0] for region in adversarial_regions)
    ).mean()
    adversarial_background = torch.stack(
        tuple(region[1] for region in adversarial_regions)
    ).mean()
    adversarial = (
        adversarial_foreground + adversarial_background
    ) * 0.5

    feature_foreground_scales = []
    feature_background_scales = []
    for fake_scale, real_scale in zip(
        fake_features,
        real_features,
        strict=True,
    ):
        feature_weight = 4.0 / (len(fake_scale) - 1)
        regions = tuple(
            _white_canvas_region_means(
                (fake.float() - real.detach().float()).abs(),
                foreground_mask,
            )
            for fake, real in zip(
                fake_scale[:-1],
                real_scale[:-1],
                strict=True,
            )
        )
        feature_foreground_scales.append(
            torch.stack(tuple(region[0] for region in regions)).sum()
            * feature_weight
        )
        feature_background_scales.append(
            torch.stack(tuple(region[1] for region in regions)).sum()
            * feature_weight
        )
    feature_matching_foreground = torch.stack(
        tuple(feature_foreground_scales)
    ).mean()
    feature_matching_background = torch.stack(
        tuple(feature_background_scales)
    ).mean()
    feature_matching = (
        feature_matching_foreground + feature_matching_background
    ) * 0.5
    reconstruction = functional.l1_loss(generated.float(), target.float())
    reconstruction_for_total = (
        reconstruction
        if reconstruction_balance == "uniform"
        else white_canvas_equal_regions_loss(generated, target)
    )
    total = (
        adversarial
        + feature_matching * lambda_feature_matching
        + reconstruction_for_total * lambda_l1
    )
    return BalancedGeneratorLoss(
        total=total,
        adversarial=adversarial,
        adversarial_foreground=adversarial_foreground,
        adversarial_background=adversarial_background,
        reconstruction=reconstruction,
        reconstruction_for_total=reconstruction_for_total,
        feature_matching=feature_matching,
        feature_matching_foreground=feature_matching_foreground,
        feature_matching_background=feature_matching_background,
    )


def white_canvas_equal_regions_loss(generated: Tensor, target: Tensor) -> Tensor:
    generated_float = generated.float()
    target_float = target.float()
    absolute_error = (generated_float - target_float).abs().mean(dim=1)
    foreground = target_float.ne(1.0).any(dim=1)
    foreground_count = foreground.sum(dim=(1, 2))
    background_count = (~foreground).sum(dim=(1, 2))
    foreground_error = (
        (absolute_error * foreground).sum(dim=(1, 2)) / foreground_count
    )
    background_error = (
        (absolute_error * ~foreground).sum(dim=(1, 2)) / background_count
    )
    return ((foreground_error + background_error) * 0.5).mean()


def white_canvas_foreground_mask(target: Tensor) -> Tensor:
    return target.float().ne(1.0).any(dim=1, keepdim=True).float()


def target_palette_proximity_loss(
    generated: Tensor,
    target: Tensor,
    target_palette: Tensor,
) -> PaletteProximityLoss:
    with torch.autocast(device_type=generated.device.type, enabled=False):
        generated_pixels = (
            generated.float()
            .permute(0, 2, 3, 1)
            .reshape(generated.shape[0], -1, generated.shape[1])
        )
        palette = target_palette.float()
        distance_map = (
            (
                generated_pixels.unsqueeze(2)
                - palette.unsqueeze(1)
            )
            .square()
            .mean(dim=3)
            .amin(dim=2)
            .view(
                generated.shape[0],
                1,
                generated.shape[2],
                generated.shape[3],
            )
        )
    foreground, background = _white_canvas_region_means(
        distance_map,
        white_canvas_foreground_mask(target),
    )
    return PaletteProximityLoss(
        total=(foreground + background) * 0.5,
        foreground=foreground,
        background=background,
    )


def _white_canvas_region_means(
    loss_map: Tensor,
    foreground_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    per_pixel = loss_map.float().mean(dim=1, keepdim=True)
    foreground = functional.adaptive_avg_pool2d(
        foreground_mask.float(),
        per_pixel.shape[-2:],
    )
    foreground_loss = (
        (per_pixel * foreground).sum(dim=(1, 2, 3))
        / foreground.sum(dim=(1, 2, 3))
    )
    background = 1.0 - foreground
    background_loss = (
        (per_pixel * background).sum(dim=(1, 2, 3))
        / background.sum(dim=(1, 2, 3))
    )
    return foreground_loss.mean(), background_loss.mean()


def model_parameter_report(
    generator: nn.Module,
    discriminator: nn.Module,
) -> dict[str, int]:
    generator_parameters = sum(parameter.numel() for parameter in generator.parameters())
    discriminator_parameters = sum(
        parameter.numel() for parameter in discriminator.parameters()
    )
    return {
        "generator": generator_parameters,
        "discriminator": discriminator_parameters,
        "total": generator_parameters + discriminator_parameters,
    }


def generator_parameter_report(generator: nn.Module) -> dict[str, int]:
    generator_parameters = sum(parameter.numel() for parameter in generator.parameters())
    return {
        "generator": generator_parameters,
        "total": generator_parameters,
    }


def _batch_norm(
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
    )
