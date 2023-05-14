from typing import Tuple

import torch


class UNet(torch.nn.Module):
    def __init__(self, input_channels: int, class_number: int, channels: int):
        super(UNet, self).__init__()

        self._contracting_layer_1 = ContractingLayer(input_channels, channels)
        self._contracting_layer_2 = ContractingLayer(channels, 2 * channels)

        self._conv = ConvLayer(2 * channels, 2 * channels)

        self._expansive_layer_1 = ExpansiveLayer(4 * channels, channels)
        self._expansive_layer_2 = ExpansiveLayer(2 * channels, channels // 2)

        self._ffd = torch.nn.Conv2d(channels // 2, class_number, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Applies UNet transformation to `x`

        :param x: Input tensor of (bs, in_channels, h, w) shape
        :return: Output tensor of (bs, h, w, class_number)
        """

        cont_x, skip_x_1 = self._contracting_layer_1(x)  # (bs, ch, h // 2, w // 2) and (bs, ch, h, w)
        cont_x, skip_x_2 = self._contracting_layer_2(cont_x)  # (bs, 2 * ch, h // 4, w // 4) and (bs, 2 * ch, h // 2, w // 2)

        x: torch.Tensor = self._conv(cont_x)  # (bs, 2 * ch, h // 4, w // 4)

        exp_x: torch.Tensor = self._expansive_layer_1(x, skip_x_2)  # (bs, ch, h // 2, w // 2)
        exp_x: torch.Tensor = self._expansive_layer_2(exp_x, skip_x_1)  # (bs, ch // 2, h, w)

        output: torch.Tensor = self._ffd(exp_x)  # (bs, class_number, h, w)
        output: torch.Tensor = output.permute(0, 2, 3, 1).contiguous()  # (bs, h, w, class_number)

        return output


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvLayer, self).__init__()
        self._conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Applies 2d convolution, batch normalization and relu activation twice

        :param x: Input tensor of (bs, in_channels, height, width) shape
        :return: Output tensor of (bs, out_channels, height, width) shape
        """
        return self._conv(x)


class ContractingLayer(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ContractingLayer, self).__init__()
        self._conv = ConvLayer(in_channels, out_channels)
        self._max_pool = torch.nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Applies a Conv transformation to 'x' and the max pools it

        :param x: Input tensor of (bs, in_channels, height, width) shape
        :return: Output 2 tensors of (bs, out_channels, height // 2, width // 2) and (bs, out_channels, height, width) shapes respectively
        """
        x = self._conv(x)
        return self._max_pool(x), x


class ExpansiveLayer(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ExpansiveLayer, self).__init__()
        self._conv = ConvLayer(in_channels, out_channels)
        self._deconv = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x: torch.Tensor, skip_x: torch.Tensor) -> torch.Tensor:
        """ Applies a Conv transformation to `x` and then upsamples it

        :param x: Input tensor of (bs, in_channels, height_1, width_1) shape
        :param skip_x: Input tensor (from one of the contracting layers) of (bs, out_channels, height_2, width_2) shape
        :return: Output tensor of (bs, out_channels, height_2, width_2) shape
        """
        x = self._deconv(x)  # (bs, in_channels, 2 * height_1, 2 * width_1)

        diff_height = skip_x.shape[2] - x.shape[2]
        diff_width = skip_x.shape[3] - x.shape[3]

        pad_size = (diff_width // 2, diff_width - diff_width // 2, diff_height // 2, diff_height - diff_height // 2)

        x = torch.nn.functional.pad(x, pad_size)  # (bs, in_channels, height_2, width_2)
        x = torch.cat([skip_x, x], dim=1)  # (bs, 2 * in_channels, height_2, width_2)

        return self._conv(x)
