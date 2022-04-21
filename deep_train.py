import numpy as np
import torch
from tqdm import tqdm
from torch import nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from steg_net import StegNet

# 定义参数
configs = {
    'host_channels': 3,
    'guest_channels': 1,
    'img_width': 300,
    'img_height': 300,
    'epoch_num': 50,
    'train_batch_size': 32,
    'val_batch_size': 32,
    'encoder_weight': 2,
    'decoder_weight': 1,
    'model_path': '/content/drive/MyDrive/MyModels/End_to_end_Stegnography_2017',
    'learning_rate': 1e-4
}

# 使用像素更高的图像集
transforms = transforms.Compose([transforms.Resize([configs['img_width'], configs['img_height']]),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                      [0.229, 0.224, 0.225])])

# Flowers102-->构建数据集
flowers102_train_dataset = datasets.Flowers102('~/.pytorch/Flowers102/', download=True, split='test',
                                               transform=transforms)
flowers102_train_data_loader = DataLoader(flowers102_train_dataset, batch_size=configs['train_batch_size'],
                                          shuffle=True)
flowers102_val_dataset = datasets.Flowers102('~/.pytorch/Flowers102/', download=True, split='val', transform=transforms)
flowers102_val_data_loader = DataLoader(flowers102_val_dataset, batch_size=configs['val_batch_size'], shuffle=True)

# Food101-->构建数据集
# food101_train_dataset = datasets.Food101('~/.pytorch/Food101/', download=True, split='train', transform=transforms)
# food101_train_data_loader = DataLoader(food101_train_dataset, batch_size=configs['train_batch_size'], shuffle=True)
# food101_val_dataset = datasets.Food101('~/.pytorch/Food101/', download=True, split='test', transform=transforms)
# food101_val_data_loader = DataLoader(food101_train_dataset, batch_size=configs['val_batch_size'], shuffle=True)

# 训练***************************************************************************************************************
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running device is {device.type}")

# 使用默认参数构建模型
model = StegNet()
model.to(device)
# model.load_model(configs['model_path'], file_name=f"steg_net"
#                                                   f"_host{configs['host_channels']}"
#                                                   f"_guest{configs['guest_channels']}"
#                                                   f"_epochNum{configs['epoch_num']}"
#                                                   f"_batchSize{configs['train_batch_size']}"
#                                                   f"_lr{configs['learning_rate']}"
#                                                   f"_encoderWeight{configs['encoder_weight']}"
#                                                   f"_decoderWeight{configs['decoder_weight']}
#                                                   f"_imgSize{configs['img_width']}x{configs['img_height']}.pth")

# 定义损失函数
criterion = nn.MSELoss()
metric = nn.L1Loss()
# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=configs['learning_rate'])
# 最小的图像损失
min_val_loss = +np.inf
# 转化灰度图
to_gray = torchvision.transforms.Grayscale(num_output_channels=1)

for epoch in range(configs['epoch_num']):
    train_loss = 0
    model.train()

    for batch in tqdm(flowers102_train_data_loader, desc=f"Flowers102 Train Epoch: {epoch}"):
        # 使用数据集的图像部分作为本模型的训练数据
        train_data, _ = [x.to(device) for x in batch]
        # 防止最后一个批次的数据不够 batch_size 大小
        if len(train_data) < configs['train_batch_size']:
            print(f"batch_size: {len(train_data)}")
            continue
        # 将一半数据作为host 另一半数据作为guest
        host_img = train_data[0:int(train_data.shape[0] / 2)]
        guest_img = train_data[int(train_data.shape[0] / 2):]
        # 使用guest的一个通道作为载密图像
        guest_img = to_gray(guest_img)

        optimizer.zero_grad()
        encoder_out, decoder_out = model(host_img, guest_img)

        # 计算均方差损失
        encoder_loss = criterion(encoder_out.view(-1, configs['img_width'] * configs['img_height']),
                                 host_img.view(-1, configs['img_width'] * configs['img_height']))
        decoder_loss = criterion(decoder_out.view(-1, configs['img_width'] * configs['img_height']),
                                 guest_img.view(-1, configs['img_width'] * configs['img_height']))
        loss = configs['encoder_weight'] * encoder_loss + configs['decoder_weight'] * decoder_loss
        loss.backward()
        optimizer.step()
        train_loss = train_loss + loss.item()

    # for batch in tqdm(food101_train_data_loader, desc=f"Food101 Train Epoch: {epoch}"):
    #     if len(batch) < configs['train_batch_size']:
    #         print(f"batch_size: {len(batch)}")
    #         continue
    #     # 使用数据集的图像部分作为本模型的训练数据
    #     train_data, _ = [x.to(device) for x in batch]
    #     # 将一半数据作为host 另一半数据作为guest
    #     host_img = train_data[0:int(train_data.shape[0] / 2)]
    #     guest_img = train_data[int(train_data.shape[0] / 2):]
    #     # 使用guest的一个通道作为载密图像
    #     guest_img = to_gray(guest_img).to(device)
    #
    #     optimizer.zero_grad()
    #     encoder_out, decoder_out = model(host_img, guest_img)
    #
    #     # 计算均方差损失
    #     encoder_loss = criterion(encoder_out.view(-1, configs['img_width'] * configs['img_height']),
    #                              host_img.view(-1, configs['img_width'] * configs['img_height']))
    #     decoder_loss = criterion(decoder_out.view(-1, configs['img_width'] * configs['img_height']),
    #                              guest_img.view(-1, configs['img_width'] * configs['img_height']))
    #     loss = configs['encoder_weight'] * encoder_loss + configs['decoder_weight'] * decoder_loss
    #     loss.backward()
    #     optimizer.step()
    #     train_loss = train_loss + loss.item()

    else:
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in tqdm(flowers102_val_data_loader, desc=f"Flowers102 Val Epoch: {epoch}"):
                # 使用数据集的图像部分作为本模型的训练数据
                val_data, _ = [x.to(device) for x in batch]
                # 防止最后一个批次的数据不够 batch_size 大小
                if len(val_data) < configs['train_batch_size']:
                    print(f"batch_size: {len(val_data)}")
                    continue
                # 将一半数据作为host 另一半数据作为guest
                host_img = val_data[0:int(val_data.shape[0] / 2)]
                guest_img = val_data[int(val_data.shape[0] / 2):]
                # 使用guest的一个通道作为载密图像
                guest_img = to_gray(guest_img)

                encoder_out, decoder_out = model(host_img, guest_img)

                # 计算均方差损失
                encoder_loss = metric(encoder_out.view(-1, configs['img_width'] * configs['img_height']),
                                      host_img.view(-1, configs['img_width'] * configs['img_height']))
                decoder_loss = metric(decoder_out.view(-1, configs['img_width'] * configs['img_height']),
                                      guest_img.view(-1, configs['img_width'] * configs['img_height']))
                loss = encoder_loss + decoder_loss
                val_loss = val_loss + loss.item()

            # for batch in tqdm(food101_val_data_loader, desc=f"Food101 Val Epoch: {epoch}"):
            #     # 使用数据集的图像部分作为本模型的训练数据
            #     val_data, _ = [x.to(device) for x in batch]
            #     # 将一半数据作为host 另一半数据作为guest
            #     host_img = val_data[0:int(val_data.shape[0] / 2)]
            #     guest_img = val_data[int(val_data.shape[0] / 2):]
            #     # 使用guest的一个通道作为载密图像
            #     guest_img = to_gray(guest_img).to(device)
            #
            #     encoder_out, decoder_out = model(host_img, guest_img)
            #
            #     # 计算均方差损失
            #     encoder_loss = metric(encoder_out.view(-1, configs['img_width'] * configs['img_height']),
            #                           host_img.view(-1, configs['img_width'] * configs['img_height']))
            #     decoder_loss = metric(decoder_out.view(-1, configs['img_width'] * configs['img_height']),
            #                           guest_img.view(-1, configs['img_width'] * configs['img_height']))
            #     loss = encoder_loss + decoder_loss
            #     val_loss = val_loss + loss.item()

    train_loss = train_loss / len(flowers102_train_data_loader)
    val_loss = val_loss / len(flowers102_val_data_loader)
    print(f"Flowers102 Epoch {epoch} train_loss: {train_loss}")
    print(f"Flowers102 Epoch {epoch} val_loss: {val_loss}")
    if val_loss <= min_val_loss:
        print(f"Validation loss decreased {min_val_loss} --> {val_loss}")
        min_val_loss = val_loss
        model.save_model(configs['model_path'], file_name=f"steg_net"
                                                          f"_host{configs['host_channels']}"
                                                          f"_guest{configs['guest_channels']}"
                                                          f"_epochNum{configs['epoch_num']}"
                                                          f"_batchSize{int(configs['train_batch_size'] / 2)}"
                                                          f"_lr{configs['learning_rate']}"
                                                          f"_encoderWeight{configs['encoder_weight']}"
                                                          f"_decoderWeight{configs['decoder_weight']}"
                                                          f"_imgSize{configs['img_width']}x{configs['img_height']}.pth")
