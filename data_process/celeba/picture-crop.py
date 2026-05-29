import os
from PIL import Image

'''裁剪原始图像，中心裁剪为指定大小的图像'''

def crop_center(image, target_size):
    width, height = image.size
    new_width, new_height = target_size

    # 计算裁剪区域的左上角和右下角坐标
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2

    # 裁剪图像
    return image.crop((left, top, right, bottom))

def batch_crop_images(input_folder, output_folder, target_size=(144, 144)):
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.jpg'):
            # 打开图片
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path)

            # 裁剪图片
            cropped_image = crop_center(image, target_size)

            # 保存裁剪后的图片
            output_path = os.path.join(output_folder, filename)
            cropped_image.save(output_path)

            print(f'Processed {filename}')

# 使用示例
input_folder = '../../data2/eval_images'  # 替换为输入文件夹路径
output_folder = '../../data2/orient'

batch_crop_images(input_folder, output_folder)
