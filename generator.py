# -*- coding:utf-8 -*-
import os
import torch
import numpy as np
from PIL import Image
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset

'''生成器网络，编码器-解码器模型（一系列卷积和批量归一化），创建匿名化图像'''

# 设置图片大小
img_size = 144
batchsz = 32
channels = 3
lr = 0.0001
b1 = 0.5
b2 = 0.999
img_shape = (channels, img_size, img_size)
image_path = './data/orient/'
transfer_path = './data/transfer/'


# 设置训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Normalize(mean=[0.500, 0.500, 0.500],
                                 std=[0.229, 0.224, 0.225])


# 加载图片
def load_img(img_path):
    img = Image.open(img_path).convert(
        'RGB')  # 使打开的图片通道为RGB格式,如果不使用.convert('RGB')进行转换的话，读出来的图像是RGBA四通道的，A通道为透明通道，该对深度学习模型训练来说暂时用不到，因此使用convert('RGB')进行通道转换。
    img = img.resize((img_size, img_size))  # 对图片进行裁剪，为144x144
    img = transforms.ToTensor()(img)
    img = transform(img).unsqueeze(0)  # unsqueeze升维，使数据格式符合[batch_size, n_channels, hight, width],[1,3,144,144]
    return img


# 显示图片
def show_img(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    return image

# 构建神经网络
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128)
        )

        self.model = nn.Sequential(
            *block(9 * 9 * 128, 2048, normalize=False),
            *block(2048, 512),
            *block(512, 128),
            *block(128, 32),
            *block(32, 128),
            *block(128, 512),
            *block(512, 2048),
            nn.Linear(2048, 9 * 9 * 128),
            nn.Sigmoid()
        )

        self.deconv_layer = nn.Sequential(
            nn.ConvTranspose2d(128,64,3,2,1,1),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 3, 3, 2, 1, 1)
        )


    def forward(self, z):
        z = self.conv1(z)
        z = z.view(z.size(0), -1)
        img = self.model(z)
        img = img.view(img.size(0), 128, 9, 9)
        img = self.deconv_layer(img)
        return img

content_img = None
transfer_img = None
for file in os.listdir(image_path):
    new_image = image_path + file
    if content_img is None:
        content_img = load_img(new_image).to(device)
    else:
        image_x = load_img(new_image).to(device)
        content_img = torch.cat([content_img, image_x], dim=0)

for file in os.listdir(transfer_path):
    new_image = transfer_path + file
    if transfer_img is None:
        transfer_img = load_img(new_image).to(device)
    else:
        image_x = load_img(new_image).to(device)
        transfer_img = torch.cat([transfer_img, image_x], dim=0)


content_img = TensorDataset(content_img)
data_loader = torch.utils.data.DataLoader(content_img, batch_size=batchsz)
transfer_img = TensorDataset(transfer_img)
label_loader = torch.utils.data.DataLoader(transfer_img, batch_size=batchsz)

generator = Generator()
generator.load_state_dict(torch.load('model_generator.pth'))
generator.to(device)
criteon = nn.MSELoss().to(device)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))

label_loader = list(label_loader)

for epoch in range(5):
    for batchid, (x) in enumerate(data_loader):
        y = label_loader[batchid]
        y = y[0].to(device)
        x = x[0].to(device)
        x = generator(x)
        loss = criteon(x, y)
        optimizer_G.zero_grad()
        loss.backward()
        optimizer_G.step()
    print(epoch, 'loss:', loss.item())
torch.save(generator.state_dict(), 'model_generator.pth')