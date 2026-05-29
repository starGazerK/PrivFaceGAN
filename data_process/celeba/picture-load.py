import os
import shutil

'''使用前面创建的类别txt文件，对原始图片数据集进行筛选分类，得到多个类别子目录'''

def copy_jpg_files(file_list_path, source_folder, destination_folder):
    # 确保目标文件夹存在
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 读取文件名列表
    with open(file_list_path, 'r', encoding='utf-8') as file:
        file_names = file.read().splitlines()

    # 遍历文件名并复制文件
    for file_name in file_names:
        # 构造源文件的完整路径
        source_file_path = os.path.join(source_folder, file_name.strip())

        # 检查文件是否存在
        if os.path.isfile(source_file_path):
            # 复制文件到目标文件夹
            shutil.copy(source_file_path, destination_folder)
            print(f'Copied: {file_name}')
        else:
            print(f'File not found: {file_name}')

# 使用示例
file_list_path = 'D:/project/face-identify/celeba/data/face_3899.txt'  # 替换为文件名列表路径
source_folder = 'D:/BaiduNetdiskDownload/Large-scale CelebFaces Attributes (CelebA) Dataset/Large-scale CelebFaces Attributes (CelebA) Dataset/Img/img_align_celeba/img_align_celeba'    # 替换为你的源文件夹路径
destination_folder = 'D:/project/face-identify/celeba/data/img_3899'  # 替换为你的目标文件夹路径

copy_jpg_files(file_list_path, source_folder, destination_folder)
