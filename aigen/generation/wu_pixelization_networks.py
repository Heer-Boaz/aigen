from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _LayerNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.empty(num_features).uniform_())
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        shape = [-1] + [1] * (value.dim() - 1)
        if value.size(0) == 1:
            mean = value.view(-1).mean().view(*shape)
            std = value.view(-1).std().view(*shape)
        else:
            mean = value.view(value.size(0), -1).mean(1).view(*shape)
            std = value.view(value.size(0), -1).std(1).view(*shape)
        normalized = (value - mean) / (std + self.eps)
        affine_shape = [1, -1] + [1] * (value.dim() - 2)
        return normalized * self.gamma.view(*affine_shape) + self.beta.view(
            *affine_shape
        )


class _ConvBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        stride: int,
        padding: int = 0,
        *,
        norm: str = "none",
        activation: str = "relu",
        pad_type: str = "zero",
    ) -> None:
        super().__init__()
        if pad_type == "reflect":
            self.pad = nn.ReflectionPad2d(padding)
        else:
            self.pad = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(
            input_dim,
            output_dim,
            kernel_size,
            stride,
            bias=True,
        )
        if norm == "in":
            self.norm: nn.Module | None = nn.InstanceNorm2d(output_dim)
        elif norm == "ln":
            self.norm = _LayerNorm(output_dim)
        else:
            self.norm = None
        if activation == "relu":
            self.activation: nn.Module | None = nn.ReLU(inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.conv(self.pad(value))
        if self.norm is not None:
            value = self.norm(value)
        if self.activation is not None:
            value = self.activation(value)
        return value


class _ResidualBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        norm: str,
        pad_type: str,
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            _ConvBlock(
                dim,
                dim,
                3,
                1,
                1,
                norm=norm,
                activation="relu",
                pad_type=pad_type,
            ),
            _ConvBlock(
                dim,
                dim,
                3,
                1,
                1,
                norm=norm,
                activation="none",
                pad_type=pad_type,
            ),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.model(value)


class _ResidualBlocks(nn.Module):
    def __init__(
        self,
        count: int,
        dim: int,
        *,
        norm: str,
        pad_type: str,
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            *(
                _ResidualBlock(dim, norm=norm, pad_type=pad_type)
                for _ in range(count)
            )
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.model(value)


class _LinearBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        activation: str,
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=True)
        self.norm = None
        self.activation = nn.ReLU(inplace=True) if activation == "relu" else None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.fc(value)
        if self.activation is not None:
            value = self.activation(value)
        return value


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            _LinearBlock(256, 256, activation="relu"),
            _LinearBlock(256, 256, activation="relu"),
            _LinearBlock(256, 256, activation="relu"),
            _LinearBlock(256, 2048, activation="none"),
        )

    def forward(self, style: torch.Tensor) -> torch.Tensor:
        flattened = style.view(style.size(0), -1)
        return self.model[3](self.model[:3](flattened))


class _ModulationConvBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, kernel_size: int) -> None:
        super().__init__()
        self.in_c = input_dim
        self.out_c = output_dim
        self.ksize = kernel_size
        self.stride = 1
        self.padding = kernel_size // 2
        self.eps = 1e-8
        self.weight = nn.Parameter(
            torch.randn(output_dim, input_dim, kernel_size, kernel_size)
        )
        self.wscale = 1.0 / math.sqrt(kernel_size * kernel_size * input_dim)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.activate_scale = math.sqrt(2.0)

    def forward(self, value: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = value.shape
        weight = self.weight * self.wscale
        weight = weight.view(1, self.ksize, self.ksize, self.in_c, self.out_c)
        weight = weight * code.view(batch, 1, 1, self.in_c, 1)
        weight_norm = torch.sqrt(torch.sum(weight**2, dim=(1, 2, 3)) + self.eps)
        weight = weight / weight_norm.view(batch, 1, 1, 1, self.out_c)
        value = value.view(1, batch * self.in_c, height, width)
        weight = weight.permute(1, 2, 3, 0, 4).reshape(
            self.ksize,
            self.ksize,
            self.in_c,
            batch * self.out_c,
        )
        weight = weight.permute(3, 2, 0, 1)
        value = F.conv2d(
            value,
            weight=weight,
            stride=self.stride,
            padding=self.padding,
            groups=batch,
        )
        value = value.view(batch, self.out_c, height, width)
        value = value + self.bias.view(1, -1, 1, 1)
        return self.activate(value) * self.activate_scale


class _RGBEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            _ConvBlock(3, 64, 7, 1, 3, norm="in", pad_type="reflect"),
            _ConvBlock(64, 128, 4, 2, 1, norm="in", pad_type="reflect"),
            _ConvBlock(128, 256, 4, 2, 1, norm="in", pad_type="reflect"),
            _ResidualBlocks(4, 256, norm="in", pad_type="reflect"),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.model(value)


class _RGBDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mod_conv_1 = _ModulationConvBlock(256, 256, 3)
        self.mod_conv_2 = _ModulationConvBlock(256, 256, 3)
        self.mod_conv_3 = _ModulationConvBlock(256, 256, 3)
        self.mod_conv_4 = _ModulationConvBlock(256, 256, 3)
        self.mod_conv_5 = _ModulationConvBlock(256, 256, 3)
        self.mod_conv_6 = _ModulationConvBlock(256, 256, 3)
        self.mod_conv_7 = _ModulationConvBlock(256, 256, 3)
        self.mod_conv_8 = _ModulationConvBlock(256, 256, 3)
        self.upsample_block1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_1 = _ConvBlock(
            256,
            128,
            5,
            1,
            2,
            norm="ln",
            pad_type="reflect",
        )
        self.upsample_block2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_2 = _ConvBlock(
            128,
            64,
            5,
            1,
            2,
            norm="ln",
            pad_type="reflect",
        )
        self.conv_3 = _ConvBlock(
            64,
            3,
            7,
            1,
            3,
            activation="tanh",
            pad_type="reflect",
        )

    def forward(self, value: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        # The released inference graph intentionally reuses mod_conv_2 here.
        residual = value
        value = self.mod_conv_1(value, code[:, :256])
        value = self.mod_conv_2(value, code[:, 256:512])
        value = value + residual
        residual = value
        value = self.mod_conv_2(value, code[:, 512:768])
        value = self.mod_conv_2(value, code[:, 768:1024])
        value = value + residual
        residual = value
        value = self.mod_conv_2(value, code[:, 1024:1280])
        value = self.mod_conv_2(value, code[:, 1280:1536])
        value = value + residual
        residual = value
        value = self.mod_conv_2(value, code[:, 1536:1792])
        value = self.mod_conv_2(value, code[:, 1792:2048])
        value = value + residual
        value = self.upsample_block1(value)
        value = self.conv_1(value)
        value = self.upsample_block2(value)
        value = self.conv_2(value)
        return self.conv_3(value)


class WuI2PNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.RGBEnc = _RGBEncoder()
        self.RGBDec = _RGBDecoder()
        self.MLP = _MLP()

    def forward(
        self,
        image: torch.Tensor,
        cell_feature: torch.Tensor,
    ) -> torch.Tensor:
        feature = self.RGBEnc(image)
        cell_code = self.MLP(cell_feature)
        return self.RGBDec(feature, cell_code)


class _AliasRGBEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            _ConvBlock(3, 64, 7, 1, 3, norm="in", pad_type="reflect"),
            _ConvBlock(64, 128, 4, 2, 1, norm="in", pad_type="reflect"),
            _ConvBlock(128, 256, 4, 2, 1, norm="in", pad_type="reflect"),
            _ResidualBlocks(3, 256, norm="in", pad_type="reflect"),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.model(value)


class _AliasRGBDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.Res_Blocks = _ResidualBlocks(3, 256, norm="in", pad_type="reflect")
        self.upsample_block1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_1 = _ConvBlock(
            256,
            128,
            5,
            1,
            2,
            norm="ln",
            pad_type="reflect",
        )
        self.upsample_block2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_2 = _ConvBlock(
            128,
            64,
            5,
            1,
            2,
            norm="ln",
            pad_type="reflect",
        )
        self.conv_3 = _ConvBlock(
            64,
            3,
            7,
            1,
            3,
            activation="tanh",
            pad_type="reflect",
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.Res_Blocks(value)
        value = self.upsample_block1(value)
        value = self.conv_1(value)
        value = self.upsample_block2(value)
        value = self.conv_2(value)
        return self.conv_3(value)


class WuAliasNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.RGBEnc = _AliasRGBEncoder()
        self.RGBDec = _AliasRGBDecoder()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.RGBDec(self.RGBEnc(value))
