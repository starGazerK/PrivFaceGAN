# -*- coding:utf-8 -*-
import os
import cv2 as cv
import torch
from PIL import Image
from torch import nn
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision import datasets

# 设置图片大小
img_size = 144
batchsz = 32
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
    img = img.resize((img_size, img_size))  # 对图片进行裁剪，为512x512
    img = transforms.ToTensor()(img)
    img = transform(img).unsqueeze(0)  # unsqueeze升维，使数据格式符合[batch_size, n_channels, hight, width],[1,3,512,512]
    return img


# 显示图片
def show_img(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    return image


# 构建神经网络
class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.select = ['0', '5', '10', '19', '28']
        self.vgg = models.vgg19(pretrained=True).features  # .features用于提取卷积层

    def forward(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)  # name为第几层的序列号，layer就是卷积层,,x为输入的图片。x = layer(x)的意思是，x经过layer层卷积后再赋值给x
            if name in self.select:
                features.append(x)
        return features


# 加载图片
style_img = load_img("style_img.jpg").to(device)
for file in os.listdir(image_path):
    new_image = image_path + file
    content_img = load_img(new_image).to(device)

    target = content_img.clone().requires_grad_(
        True)  # clone()操作后的tensor requires_grad=True，clone操作在不共享数据内存的同时支持梯度梯度传递与叠加
    optimizer = torch.optim.Adam([target], lr=0.003)  # 选择优化器
    vgg = VGGNet().to(device).eval()
    total_step = 6000  # 训练次数
    style_weight = 90  # 给style_loss加上的权重

    content_features = [x.detach() for x in vgg(content_img)]
    style_features = [x.detach() for x in vgg(style_img)]

    # 开始训练
    for step in range(total_step):
        target_features = vgg(target)

        style_loss = 0
        content_loss = 0
        for f1, f2, f3 in zip(target_features, content_features, style_features):
            content_loss = torch.mean((f1 - f2) ** 2) + content_loss
            _, c, h, w = f1.size()  # 结果为torch.Size([1, 64, 512, 512])
            f1 = f1.view(c, h * w)  # 处理数据格式为后面gram计算
            f3 = f3.view(c, h * w)

            # 计算gram matrix
            f1 = torch.mm(f1, f1.t())  # torch.mm()两个矩阵相乘,.t()是矩阵倒置
            f3 = torch.mm(f3, f3.t())
            style_loss = torch.mean((f1 - f3) ** 2) / (c * h * w) + style_loss

        loss = content_loss + style_weight * style_loss

        # 更新target
        optimizer.zero_grad()  # 每一次优化都要梯度清零
        loss.backward()  # 反向传播
        optimizer.step()
        denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
        img = target.clone().squeeze()  # 升维-->降维
        img = denorm(img).clamp_(0, 1)
        img = show_img(img)
        print("Step [{}/{}], Content Loss: {:.4f}, Style Loss: {:.4f}"
              .format(step, total_step, content_loss.item(), style_loss.item()))
    img = img.permute(1, 2, 0)
    scr = img.detach().numpy()
    new_image = transfer_path + file
    scr = cv.normalize(scr, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    cv.imwrite(new_image, scr)
