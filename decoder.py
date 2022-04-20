import torch
from torch import nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, host_channels=3, guest_channels=1, kernel_size=3):
        """
        使用二维卷积定义解码器
        :param in_channels: 输入图像通道数
        :param out_channels: 输出图像通道数
        :param kernel_size: 编码器二维卷积核大小（方形）
        """
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=host_channels, out_channels=16, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=kernel_size, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=kernel_size, padding=1)
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=kernel_size, padding=1)
        self.conv5 = nn.Conv2d(in_channels=8, out_channels=3, kernel_size=kernel_size, padding=1)
        self.conv6 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=kernel_size, padding=1)
        self.conv7 = nn.Conv2d(in_channels=3, out_channels=guest_channels, kernel_size=kernel_size, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.conv7(x)

        return x


if __name__ == '__main__':
    decoder = Decoder()
    input = torch.ones(1, 3, 300, 300)
    output = decoder(input)
    print(output.shape)
