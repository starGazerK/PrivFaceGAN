import os
import shutil
from multiprocessing import Pool, cpu_count
from functools import partial
import time


def create_label_files_and_get_filenames(annotation_file_path, target_ids, label_file_directory):
    """
    扫描主注释文件，为每个目标ID创建独立的类别txt文件，并返回所有找到的图片文件名的集合。

    Args:
        annotation_file_path (str): 'identity_CelebA.txt' 文件的完整路径。
        target_ids (list of str): 您希望查找的类别ID列表。
        label_file_directory (str): 用于存放生成的 face_xxxx.txt 文件的目录。

    Returns:
        set: 包含所有目标类别图片文件名的唯一集合。
    """
    print(f"确保标签文件目录存在: {label_file_directory}")
    os.makedirs(label_file_directory, exist_ok=True)

    print("开始扫描主注释文件...")
    # 创建一个字典，用于按类别ID存储文件名
    category_files = {id: [] for id in target_ids}

    try:
        with open(annotation_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    image_name, image_id = parts
                    if image_id in category_files:
                        category_files[image_id].append(image_name)
    except FileNotFoundError:
        print(f"错误：注释文件未找到，路径: {annotation_file_path}")
        return set()

    # 写入独立的类别.txt文件，并收集所有文件名
    all_filenames = set()
    for cat_id, filenames in category_files.items():
        if not filenames:
            print(f"警告：类别ID '{cat_id}' 未找到任何图片。")
            continue

        output_path = os.path.join(label_file_directory, f'face_{cat_id}.txt')
        with open(output_path, 'w', encoding='utf-8') as outfile:
            outfile.write('\n'.join(filenames))

        print(f"已为类别 '{cat_id}' 创建标签文件，包含 {len(filenames)} 个图片，存放于: {output_path}")

        # 将当前类别的所有文件名添加到总集合中
        all_filenames.update(filenames)

    print(f"\n总共找到 {len(all_filenames)} 个唯一的图片文件进行复制。")
    return all_filenames


def copy_worker(filename, source_folder, destination_folder):
    """
    多进程复制文件的具体执行单元。

    Args:
        filename (str): 要复制的文件名。
        source_folder (str): 原始图片所在的源文件夹。
        destination_folder (str): 复制操作的目标文件夹。

    Returns:
        str: 被成功处理的文件名，如果文件未找到则返回None。
    """
    source_path = os.path.join(source_folder, filename)
    if os.path.exists(source_path):
        shutil.copy(source_path, destination_folder)
        return filename
    else:
        # 如果需要看到哪些文件没找到，可以取消下面这行的注释
        # print(f"警告: 源文件未找到，无法复制: {filename}")
        return None


def main():
    """
    主函数，用于组织和执行所有操作。
    """
    # --- 用户配置区 ---
    # 1. 您希望提取的类别ID列表 (请以字符串形式提供)
    TARGET_CATEGORY_IDS = ['2937', '6369', '3332', '612', '1854', '9290', '1', '4883']  # 在此添加或修改您需要的类别ID

    # 2. CelebA主注释文件的路径
    ANNOTATION_FILE = 'D:/BaiduNetdiskDownload/Large-scale CelebFaces Attributes (CelebA) Dataset/Large-scale CelebFaces Attributes (CelebA) Dataset/Anno/identity_CelebA.txt'

    # 3. 存放生成的 face_xxxx.txt 标签文件的目录
    LABEL_FILE_DIRECTORY = '../data2/label'  # 推荐新建一个专门存放标签文件的目录

    # 4. 原始图片所在的源文件夹路径
    SOURCE_IMAGE_FOLDER = 'D:/BaiduNetdiskDownload/Large-scale CelebFaces Attributes (CelebA) Dataset/Large-scale CelebFaces Attributes (CelebA) Dataset/Img/img_align_celeba/img_align_celeba'

    # 5. 所有图片最终要复制到的总目标文件夹路径
    DESTINATION_FOLDER = '../data2/eval_images'
    # --- 配置区结束 ---

    # 确保最终存放图片的目标目录存在
    if not os.path.exists(DESTINATION_FOLDER):
        os.makedirs(DESTINATION_FOLDER)
        print(f"已创建总图片目标目录: {DESTINATION_FOLDER}")

    # --- 步骤 1: 创建各类别标签文件(.txt)并获取所有需要复制的文件名 ---
    filenames_to_copy = create_label_files_and_get_filenames(ANNOTATION_FILE, TARGET_CATEGORY_IDS, LABEL_FILE_DIRECTORY)

    if not filenames_to_copy:
        print("未找到任何需要复制的文件，程序退出。")
        return

    # --- 步骤 2: 使用多进程并行复制文件 ---
    print(f"\n准备使用 {cpu_count()-8} 个CPU核心进行并行复制...")
    start_time = time.time()

    # 使用 functools.partial 来“预设”复制函数的固定参数（源和目标文件夹）
    copy_func = partial(copy_worker, source_folder=SOURCE_IMAGE_FOLDER, destination_folder=DESTINATION_FOLDER)

    # 创建进程池
    with Pool(cpu_count()) as pool:
        # 使用 map 方法将复制任务分配给进程池
        # 列表推导式会过滤掉那些因文件未找到而返回None的结果
        results = [result for result in pool.map(copy_func, filenames_to_copy) if result is not None]

    end_time = time.time()

    print("\n--- ✅ 处理完成 ---")
    print(f"成功复制了 {len(results)} 个图片文件。")
    print(f"总耗时: {end_time - start_time:.2f} 秒。")
    print(f"所有图片均已存放于: {DESTINATION_FOLDER}")
    print(f"独立的类别标签文件存放于: {LABEL_FILE_DIRECTORY}")


if __name__ == '__main__':
    main()