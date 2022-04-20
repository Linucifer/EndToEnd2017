import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, host_channels=3, guest_channels=1, kernel_size=3):
        """
        使用二维卷积定义编码器
        :param host_channels: 载体图像通道数
        :param guest_channels: 载密图像通道数
        :param kernel_size:  图像融合阶段使用的卷积核大小（方形）
        """
        super(Encoder, self).__init__()
        # host
        self.conv_h1 = nn.Conv2d(in_channels=host_channels, out_channels=16, kernel_size=kernel_size, padding=1)
        self.conv_h2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=kernel_size, padding=1)
        self.conv_h3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=kernel_size, padding=1)
        self.conv_h4 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=kernel_size, padding=1)
        self.conv_h5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=kernel_size, padding=1)
        self.conv_h6 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=kernel_size, padding=1)
        self.conv_h7 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=kernel_size, padding=1)
        self.conv8 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1)
        self.conv9 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1)
        self.conv10 = nn.Conv2d(in_channels=8, out_channels=host_channels, kernel_size=1)

        # guest
        self.conv_g1 = nn.Conv2d(in_channels=guest_channels, out_channels=16, kernel_size=kernel_size, padding=1)
        self.conv_g2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=kernel_size, padding=1)
        self.conv_g3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=kernel_size, padding=1)
        self.conv_g4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=kernel_size, padding=1)
        self.conv_g5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=kernel_size, padding=1)
        self.conv_g6 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=kernel_size, padding=1)
        self.conv_g7 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=kernel_size, padding=1)

    def forward(self, host, payload):
        h = F.relu(self.conv_h1(host))
        g = F.relu(self.conv_g1(payload))
        h = torch.cat((h, g), 1)

        h = F.relu(self.conv_h2(h))
        g = F.relu(self.conv_g2(g))

        h = F.relu(self.conv_h3(h))
        g = F.relu(self.conv_g3(g))
        h = torch.cat((h, g), 1)

        h = F.relu(self.conv_h4(h))
        g = F.relu(self.conv_g4(g))

        h = F.relu(self.conv_h5(h))
        g = F.relu(self.conv_g5(g))
        h = torch.cat((h, g), 1)

        h = F.relu(self.conv_h6(h))
        g = F.relu(self.conv_g6(g))

        h = F.relu(self.conv_h7(h))
        g = F.relu(self.conv_g7(g))
        h = torch.cat((h, g), 1)

        h = F.relu(self.conv8(h))
        h = F.relu(self.conv9(h))
        h = self.conv10(h)

        return h


if __name__ == '__main__':
    encoder = Encoder()
    host = torch.ones(1, 3, 300, 300)
    payload = torch.ones(1, 1, 300, 300)
    output = encoder(host, payload)
    print(output.shape)
