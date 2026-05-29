# -*- coding:utf-8 -*-
import os
import cv2 as cv
import torch
from PIL import Image
from torch import nn
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import time


def main():
    # --- 用户配置区 ---
    IMG_SIZE = 144  # 设置图片大小
    BATCH_SIZE = 32  # 批处理大小，根据你的GPU显存调整。如果遇到显存不足(out of memory)的错误，请减小此数值
    CONTENT_IMAGE_PATH = '../../data2/orient/'  # 存放待处理图片(内容图片)的目录
    STYLE_IMAGE_PATH = 'style_img.jpg'  # 风格图片的路径
    TRANSFER_RESULT_PATH = '../../data2/transfer2/'  # 迁移后图片的保存目录

    TOTAL_STEP = 3000  # 训练步数。由于批量处理效率更高，可以适当减少步数，1500-2000步通常效果就不错
    STYLE_WEIGHT = 60  # 风格损失的权重
    LEARNING_RATE = 0.003  # 学习率
    # --- 配置区结束 ---

    # 1. 设置运行设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")
    os.makedirs(TRANSFER_RESULT_PATH, exist_ok=True)

    # 2. 定义图片预处理流程
    preprocess = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.500, 0.500, 0.500],
                             std=[0.229, 0.224, 0.225])
    ])

    # 3. 使用ImageFolder和DataLoader进行批量加载
    content_dataset = ImageFolder(CONTENT_IMAGE_PATH, transform=preprocess)
    content_loader = DataLoader(content_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"成功加载 {len(content_dataset)} 张内容图片，将分为 {len(content_loader)} 个批次处理。")

    # 4. 构建并加载VGG特征提取网络 (只执行一次)
    class VGGNet(nn.Module):
        def __init__(self):
            super(VGGNet, self).__init__()
            self.select = ['0', '5', '10', '19', '28']
            # 【代码优化】使用 'weights' 参数替换已弃用的 'pretrained'
            self.vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features

        def forward(self, x):
            features = []
            for name, layer in self.vgg._modules.items():
                x = layer(x)
                if name in self.select:
                    features.append(x)
            return features

    vgg = VGGNet().to(device).eval()
    print("VGG19特征提取网络加载完成。")

    # 5. 加载风格图片并预计算其特征 (只执行一次)
    style_img_pil = Image.open(STYLE_IMAGE_PATH).convert('RGB')
    style_img = preprocess(style_img_pil).unsqueeze(0).to(device)
    style_features = [feature.detach() for feature in vgg(style_img)]
    print("风格图片特征计算完成。")

    # ---------------- 开始批量处理 ----------------
    start_time = time.time()
    for batch_idx, (content_batch, _) in enumerate(content_loader):
        batch_start_time = time.time()
        print(f"\n--- 正在处理第 {batch_idx + 1}/{len(content_loader)} 批图片 ---")

        content_batch = content_batch.to(device)
        content_features = [feature.detach() for feature in vgg(content_batch)]
        target_batch = content_batch.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([target_batch], lr=LEARNING_RATE)

        for step in range(TOTAL_STEP):
            target_features = vgg(target_batch)

            content_loss = 0
            style_loss = 0

            for t_feat, c_feat, s_feat in zip(target_features, content_features, style_features):
                content_loss += torch.mean((t_feat - c_feat) ** 2)

                b, c, h, w = t_feat.size()

                # Reshape目标特征图用于计算Gram矩阵
                t_feat_reshaped = t_feat.view(b, c, h * w)

                # 【【【
                #   核心错误修复点：
                #   s_feat 的原始维度是 [1, c, h, w]，需要先扩展到 [b, c, h, w]
                #   expand_as(t_feat) 可以高效地将其扩展为与 t_feat 完全相同的维度
                # 】】】
                s_feat_expanded = s_feat.expand_as(t_feat)
                s_feat_reshaped = s_feat_expanded.view(b, c, h * w)

                # 计算Gram矩阵: (b, c, hw) * (b, hw, c) -> (b, c, c)
                t_gram = torch.bmm(t_feat_reshaped, t_feat_reshaped.transpose(1, 2))
                s_gram = torch.bmm(s_feat_reshaped, s_feat_reshaped.transpose(1, 2))

                style_loss += torch.mean((t_gram - s_gram) ** 2) / (c * h * w)

            total_loss = content_loss + STYLE_WEIGHT * style_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (step + 1) % 500 == 0:
                print(f"批次 [{batch_idx + 1}/{len(content_loader)}], 步骤 [{step + 1}/{TOTAL_STEP}], "
                      f"内容损失: {content_loss.item():.4f}, 风格损失: {style_loss.item():.4f}")

        # --- 批次处理完成，保存图片 ---
        denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
        paths = [s[0] for s in content_loader.dataset.samples[BATCH_SIZE * batch_idx: BATCH_SIZE * (batch_idx + 1)]]

        for i in range(target_batch.size(0)):
            img_tensor = target_batch[i].clone().squeeze()
            img_denorm = denorm(img_tensor).clamp_(0, 1)
            img_np = img_denorm.cpu().detach().permute(1, 2, 0).numpy()
            img_np = cv.normalize(img_np, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
            img_np_bgr = cv.cvtColor(img_np, cv.COLOR_RGB2BGR)
            original_filename = os.path.basename(paths[i])
            save_path = os.path.join(TRANSFER_RESULT_PATH, original_filename)
            cv.imwrite(save_path, img_np_bgr)

        batch_time = time.time() - batch_start_time
        print(f"批次 {batch_idx + 1} 处理完毕，耗时 {batch_time:.2f} 秒。图片已保存。")

    total_time = time.time() - start_time
    print(f"\n--- ✅ 所有图片处理完成 ---")
    print(f"总耗时: {total_time:.2f} 秒。")


if __name__ == '__main__':
    main()