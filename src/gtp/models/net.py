import math

import torch
import torch.nn as nn


def calc_output_shape(
    in_shape,
    kernel_shape,
    padding_shape=(0, 0),
    stride_shape=(1, 1),
    dialation_shape=(1, 1),
):
    def get_out_size(idx):
        out_size = in_shape[idx] + 2 * padding_shape[idx]
        out_size -= dialation_shape[idx] * (kernel_shape[idx] - 1)
        out_size -= 1
        out_size /= stride_shape[idx]
        out_size = math.floor(out_size + 1)
        return out_size

    h_out = get_out_size(0)
    w_out = get_out_size(1)

    return h_out, w_out


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        inner_channels=None,
        kernel_size=3,
        dropout=False,
    ):
        super().__init__()

        if inner_channels is None:
            inner_channels = in_channels

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

        if dropout:
            self.dropout_module = nn.Dropout(0.75)
        else:
            self.dropout_module = nn.Identity()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                inner_channels,
                kernel_size=(kernel_size, 1),
                stride=1,
                padding="same",
            ),
            nn.BatchNorm2d(inner_channels),
            nn.Conv2d(
                inner_channels,
                out_channels,
                kernel_size=(kernel_size, 1),
                stride=1,
                padding="same",
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x = self.dropout_module(x)
        h = self.block(x)
        sc = self.skip(x)
        return h + sc


class LargeNet(nn.Module):
    def __init__(self, num_out_dims=10):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=0, stride=1)
        self.norm = nn.BatchNorm2d(4)
        self.act = nn.LeakyReLU(0.2)

        pool_stride = 16

        self.block1 = ResBlock(4, 8, kernel_size=20, dropout=False)
        self.pool1 = nn.AvgPool2d((pool_stride, 1), stride=(pool_stride, 1), padding=0)
        self.block2 = ResBlock(8, 8, kernel_size=20, dropout=False)
        self.pool2 = nn.MaxPool2d((pool_stride, 1), stride=(pool_stride, 1), padding=0)
        self.block3 = ResBlock(8, 16, kernel_size=20, dropout=False)
        self.pool3 = nn.MaxPool2d((pool_stride, 1), stride=(pool_stride, 1), padding=0)
        self.block4 = ResBlock(16, 16, kernel_size=20, dropout=False)

        self.dropout_module = nn.Identity()  # nn.Dropout(0.75)

        out_size = (20, 1)
        self.adpt_avg_pool = nn.AdaptiveAvgPool2d(out_size)

        self.num_features = out_size[0] * out_size[1] * 16
        self.fc = nn.Linear(self.num_features, num_out_dims)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.pool3(x)
        x = self.block4(x)
        x = self.adpt_avg_pool(x)
        x = self.dropout_module(x)
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        x = self.fc(x)
        return x


class SoyBeanNetDeep(nn.Module):
    def __init__(
        self,
        window_size=200,
        num_out_dims=10,
        insize=4,
        hidden_dim=10,
        drop_out_prob=0.75,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=(4, insize), padding=0, stride=1),
            nn.Tanh(),
            nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size=(20, 1), padding="same", stride=1
            ),
            nn.Tanh(),
        )

        block_out_size = calc_output_shape(
            in_shape=(window_size, insize),
            kernel_shape=(4, insize),
            padding_shape=(0, 0),
            stride_shape=(1, 1),
        )
        block_out_size = calc_output_shape(
            in_shape=block_out_size,
            kernel_shape=(20, insize),
            padding_shape=(20 - 1, 0),
            stride_shape=(1, 1),
        )
        self.dropout_block = nn.Dropout(drop_out_prob)

        self.shortcut = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=(4, insize), padding=0, stride=1),
            nn.Tanh(),
        )

        shortcut_out_size = calc_output_shape(
            in_shape=(window_size, insize),
            kernel_shape=(4, insize),
            padding_shape=(0, 0),
            stride_shape=(1, 1),
        )

        self.last_block = nn.Sequential(
            nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size=(100, 1), padding=0, stride=(100, 1)
            ),
            nn.Tanh(),
            nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size=(100, 1), padding=0, stride=(100, 1)
            ),
            nn.Tanh(),
            nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size=(100, 1), padding=0, stride=(100, 1)
            ),
        )
        final_out_size = calc_output_shape(
            in_shape=shortcut_out_size,
            kernel_shape=(100, 1),
            padding_shape=(0, 0),
            stride_shape=(100, 1),
        )
        final_out_size = calc_output_shape(
            in_shape=final_out_size,
            kernel_shape=(100, 1),
            padding_shape=(0, 0),
            stride_shape=(100, 1),
        )
        final_out_size = calc_output_shape(
            in_shape=final_out_size,
            kernel_shape=(100, 1),
            padding_shape=(0, 0),
            stride_shape=(100, 1),
        )

        self.dropout_last_block = nn.Dropout(drop_out_prob)

        num_features = final_out_size[0] * final_out_size[1] * hidden_dim
        self.fc = nn.Linear(num_features, num_out_dims)

    def forward(self, x):
        h = self.block(x)
        h = self.dropout_block(h)
        sc = self.shortcut(x)
        h = self.last_block(h + sc)
        h = self.dropout_last_block(h)
        h = h.view(h.shape[0], -1)
        return self.fc(h)


class SoyBeanNet(nn.Module):
    def __init__(
        self,
        window_size=200,
        num_out_dims=10,
        insize=4,
        hidden_dim=10,
        drop_out_prob=0.75,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=(4, insize), padding=0, stride=1),
            nn.Tanh(),
            nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size=(20, 1), padding="same", stride=1
            ),
            nn.Tanh(),
        )
        self.dropout_block = nn.Dropout(drop_out_prob)

        self.shortcut = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=(4, insize), padding=0, stride=1),
            nn.Tanh(),
        )

        self.last_block = nn.Sequential(
            nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size=(4, 1), padding="same", stride=1
            )
        )

        self.dropout_last_block = nn.Dropout(drop_out_prob)

        num_features = (window_size - 3) * hidden_dim
        self.fc = nn.Linear(num_features, num_out_dims)

    def forward(self, x):
        h = self.block(x)
        h = self.dropout_block(h)
        sc = self.shortcut(x)
        h = self.last_block(h + sc)
        h = self.dropout_last_block(h)
        h = h.view(h.shape[0], -1)
        return self.fc(h)


class SoyBeanNetLarge(nn.Module):
    def __init__(
        self,
        window_size=200,
        num_out_dims=10,
        insize=4,
        hidden_dim=10,
        drop_out_prob=0.75,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=(4, insize), padding=0, stride=1),
            nn.Tanh(),
            nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size=(20, 1), padding="same", stride=1
            ),
            nn.Tanh(),
        )
        self.dropout_block = nn.Dropout(drop_out_prob)

        self.shortcut = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=(4, insize), padding=0, stride=1),
            nn.Tanh(),
        )

        self.last_block = nn.Sequential(
            nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size=(4, 1), padding="same", stride=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size=(4, 1), padding="same", stride=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size=(4, 1), padding="same", stride=1
            ),
        )

        self.dropout_last_block = nn.Dropout(drop_out_prob)

        num_features = (window_size - 3) * hidden_dim
        self.fc = nn.Linear(num_features, num_out_dims)

    def forward(self, x):
        h = self.block(x)
        h = self.dropout_block(h)
        sc = self.shortcut(x)
        h = self.last_block(h + sc)
        h = self.dropout_last_block(h)
        h = h.view(h.shape[0], -1)
        return self.fc(h)


class ConvNet(nn.Module):
    def __init__(self, num_out_dims=10):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=0, stride=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((2, 1), stride=(2, 1), padding=0),
            nn.Conv2d(8, 8, kernel_size=(4, 1), padding=0, stride=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((2, 1), stride=(2, 1), padding=0),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(4, 1), padding=0, stride=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((2, 1), stride=(2, 1), padding=0),
            nn.Conv2d(16, 16, kernel_size=(4, 1), padding=0, stride=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((2, 1), stride=(2, 1), padding=0),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(4, 1), padding=0, stride=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((2, 1), stride=(2, 1), padding=0),
            nn.Conv2d(16, 16, kernel_size=(4, 1), padding=0, stride=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((2, 1), stride=(2, 1), padding=0),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(4, 1), padding=0, stride=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((2, 1), stride=(2, 1), padding=0),
            nn.Conv2d(16, 16, kernel_size=(4, 1), padding=0, stride=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((2, 1), stride=(2, 1), padding=0),
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(4, 1), padding=0, stride=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((2, 1), stride=(2, 1), padding=0),
            nn.Conv2d(16, 16, kernel_size=(4, 1), padding=0, stride=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((2, 1), stride=(2, 1), padding=0),
        )

        self.fc = nn.Linear(3392 // 4, num_out_dims)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.view(len(x), -1)
        # print(x.shape)
        x = self.fc(x)
        return x


class DeepNet(nn.Module):
    def __init__(
        self,
        window_size=200,
        num_out_dims=10,
        insize=4,
        hidden_dim=10,
        drop_out_prob=0.75,
    ):
        super().__init__()

        self.in_stride = 100
        self.in_block = nn.Sequential(
            nn.Conv2d(
                1,
                16,
                kernel_size=(self.in_stride, insize),
                padding=0,
                stride=self.in_stride,
            ),
            nn.Tanh(),
        )

        length_of_feature = calc_feature_size(
            input_size=window_size,
            kernel_size=self.in_stride,
            padding=0,
            dialation=1,
            stride=self.in_stride,
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                16,
                32,
                kernel_size=(100, 1),
                padding="same",
                stride=1,
            ),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                32,
                32,
                kernel_size=(100, 1),
                padding=0,
                stride=100,
            ),
            nn.LeakyReLU(0.2),
        )

        length_of_feature = calc_feature_size(
            input_size=length_of_feature,
            kernel_size=100,
            padding=0,
            dialation=1,
            stride=100,
        )

        num_features = length_of_feature * 32
        self.fc = nn.Linear(num_features, num_out_dims)

    def forward(self, x):
        h = self.in_block(x)
        h = self.layer1(h)
        h = torch.flatten(h, 1)
        return self.fc(h)


def calc_feature_size(input_size, kernel_size, padding, dialation, stride):
    ps = 2 * padding
    ds = dialation * (kernel_size - 1)
    numerator = input_size + ps - ds - 1
    denominator = stride
    return math.floor(numerator / denominator + 1)


class DeepNet2(nn.Module):
    def __init__(self, window_size=200, num_out_dims=10, insize=4):
        super().__init__()

        kernel_length = 100
        stride = kernel_length // 2

        self.in_block = nn.Sequential(
            nn.Conv2d(
                1,
                16,
                kernel_size=(kernel_length, insize),
                padding=0,
                stride=stride,
            ),
            nn.LeakyReLU(0.2),
        )

        length_of_feature = calc_feature_size(
            input_size=window_size,
            kernel_size=kernel_length,
            padding=0,
            dialation=1,
            stride=stride,
        )

        layer_configs = [
            (16, 32, 50, 10),
            (32, 64, 20, 2),
            (64, 64, 20, 2),
            (64, 64, 10, 2),
            (64, 32, 3, 2),
        ]
        modules = []
        for in_channels, out_channels, kernel_size, stride in layer_configs:
            modules.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(kernel_size, 1),
                    padding=0,
                    stride=stride,
                )
            )
            modules.append(nn.LeakyReLU(0.2))

            length_of_feature = calc_feature_size(
                input_size=length_of_feature,
                kernel_size=kernel_size,
                padding=0,
                dialation=1,
                stride=stride,
            )

        self.layer1 = nn.Sequential(*modules)

        num_features = length_of_feature * layer_configs[-1][1]
        self.fc = nn.Sequential(
            nn.Linear(num_features, 500),
            nn.ReLU(),
            nn.Linear(500, num_out_dims),
        )

    def forward(self, x):
        h = self.in_block(x)
        h = self.layer1(h)
        h = torch.flatten(h, 1)
        return self.fc(h)
