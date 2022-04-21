import os

import torch
from torch import nn
from decoder import Decoder
from encoder import Encoder


class StegNet(nn.Module):
    def __init__(self, host_channels=3, guest_channels=1, kernel_size=(3,3)):
        """
        使用定义好的encoder和decoder构造模型
        :param host_channels: 载体图像通道数
        :param guest_channels: 载密图像通道数
        :param kernel_size:  图像融合阶段和解码使用的卷积核大小（方形）
        """
        super(StegNet, self).__init__()
        self.encoder = Encoder(host_channels=host_channels, guest_channels=guest_channels, kernel_size=kernel_size)
        self.decoder = Decoder(host_channels=host_channels, guest_channels=guest_channels, kernel_size=kernel_size)

    def forward(self, h, g):
        encoder_out = self.encoder(h, g)
        decoder_out = self.decoder(encoder_out)

        return encoder_out, decoder_out

    def save_model(self, path, file_name='steg_net_host3_guest1.pth'):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, file_name))
        print(f"Save model successfully! path: {os.path.join(path, file_name)}")

    def load_model(self, path, file_name='steg_net_host3_guest1.pth'):
        self.load_state_dict(torch.load(os.path.join(path, file_name)))
        print(f"Load model successfully! path: {os.path.join(path, file_name)}")


if __name__ == '__main__':
    model = StegNet()
    host = torch.ones(1, 3, 300, 300)
    payload = torch.ones(1, 1, 300, 300)
    encoder_out, decoder_out = model(host, payload)
    print(encoder_out.shape)
    print(decoder_out.shape)
