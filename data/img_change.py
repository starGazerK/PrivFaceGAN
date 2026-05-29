import os
import shutil

def copy_matching_files(folder_a, folder_b, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取文件夹A中的所有文件名
    files_in_a = set(os.listdir(folder_a))

    # 遍历文件夹B中的文件
    for filename in os.listdir(folder_b):
        # 如果文件名在文件夹A中存在，则复制文件
        if filename in files_in_a:
            source_file = os.path.join(folder_b, filename)
            destination_file = os.path.join(output_folder, filename)
            shutil.copy2(source_file, destination_file)  # 使用copy2保留元数据
            print(f'Copied: {filename} to {output_folder}')

# 文件夹路径
folder_a = 'D:/project/face-identity/pre_img'  # 替换为文件夹A的路径
folder_b = 'D:/project/face-identity/transfer'  # 替换为文件夹B的路径
output_folder = 'D:/project/face-identity/new_img'  # 替换为输出文件夹的路径

# 调用函数
copy_matching_files(folder_a, folder_b, output_folder)
