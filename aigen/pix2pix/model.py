from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from aigen.pix2pix.config import ModelConfig


NormalizationFactory = Callable[[int], nn.Module]


class Pix2PixGenerator(nn.Module):
    """Reference pix2pix U-Net generator for a power-of-two square canvas."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        norm = _batch_norm
        channels = config.generator_channels
        block = _UnetBlock(
            channels * 8,
            channels * 8,
            innermost=True,
            norm=norm,
        )
        num_downs = config.image_size.bit_length() - 1
        for _ in range(num_downs - 5):
            block = _UnetBlock(
                channels * 8,
                channels * 8,
                child=block,
                norm=norm,
                dropout=config.generator_dropout,
            )
        block = _UnetBlock(channels * 4, channels * 8, child=block, norm=norm)
        block = _UnetBlock(channels * 2, channels * 4, child=block, norm=norm)
        block = _UnetBlock(channels, channels * 2, child=block, norm=norm)
        self.network = _UnetBlock(
            config.output_channels,
            channels,
            input_channels=config.input_channels,
            child=block,
            outermost=True,
            norm=norm,
        )

    def forward(self, source: Tensor) -> Tensor:
        return self.network(source)


class ConditionalPatchDiscriminator(nn.Module):
    """Reference conditional N-layer PatchGAN discriminator."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        input_channels = config.input_channels + config.output_channels
        channels = config.discriminator_channels
        layers: list[nn.Module] = [
            nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
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
                    ),
                    nn.BatchNorm2d(channels * multiplier),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
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
                ),
                nn.BatchNorm2d(channels * multiplier),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(
                    channels * multiplier,
                    1,
                    kernel_size=4,
                    stride=1,
                    padding=1,
                ),
            ]
        )
        self.network = nn.Sequential(*layers)

    def forward(self, source: Tensor, candidate: Tensor) -> Tensor:
        return self.network(torch.cat((source, candidate), dim=1))


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
        nn.init.normal_(module.weight, mean=1.0, std=0.02)
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


def discriminator_loss(real_logits: Tensor, fake_logits: Tensor) -> Tensor:
    real = functional.softplus(-real_logits.float()).mean()
    fake = functional.softplus(fake_logits.float()).mean()
    return (real + fake) * 0.5


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


def _batch_norm(channels: int) -> nn.Module:
    return nn.BatchNorm2d(channels, affine=True, track_running_stats=True)
