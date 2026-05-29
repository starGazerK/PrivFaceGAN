import os
import shutil
import random


def split_dataset(base_path='.'):
    """
    根据label文件划分数据集为训练集和测试集，并移动相应的图片文件。

    Args:
        base_path (str): 脚本运行时，所有相关目录所在的根目录。
                         默认为当前目录。
    """
    # --- 1. 定义所有需要的目录路径 ---
    label_dir = os.path.join(base_path, 'label')
    orient_dir = os.path.join(base_path, 'orient')
    transfer_dir = os.path.join(base_path, 'transfer')

    label_train_dir = os.path.join(base_path, 'label-train')
    label_test_dir = os.path.join(base_path, 'label-test')
    orient_test_dir = os.path.join(base_path, 'orient-1')
    transfer_test_dir = os.path.join(base_path, 'transfer-1')

    # 定义每个类别的测试集样本数量
    NUM_TEST_SAMPLES = 5

    # --- 2. 创建所有输出目录（如果不存在）---
    print("正在创建输出目录...")
    os.makedirs(label_train_dir, exist_ok=True)
    os.makedirs(label_test_dir, exist_ok=True)
    os.makedirs(orient_test_dir, exist_ok=True)
    os.makedirs(transfer_test_dir, exist_ok=True)
    print("输出目录创建完成。")

    # --- 3. 遍历label目录下的所有txt文件 ---
    try:
        # 获取所有以.txt结尾的文件，避免处理其他无关文件
        category_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    except FileNotFoundError:
        print(f"错误：输入标签目录 '{label_dir}' 不存在！请检查路径。")
        return

    if not category_files:
        print(f"警告：在 '{label_dir}' 目录中未找到任何 .txt 标签文件。")
        return

    print(f"\n找到 {len(category_files)} 个类别标签文件，开始处理...")

    # --- 4. 处理每个类别文件 ---
    for category_filename in category_files:
        print(f"\n--- 正在处理类别: {category_filename} ---")

        # 读取当前类别下的所有图片文件名
        source_label_path = os.path.join(label_dir, category_filename)
        with open(source_label_path, 'r', encoding='utf-8') as f:
            # 使用列表推导式过滤掉空行
            all_files = [line.strip() for line in f if line.strip()]

        if not all_files:
            print("警告：该标签文件为空，已跳过。")
            continue

        # 检查样本数量是否足够划分
        if len(all_files) > NUM_TEST_SAMPLES:
            num_to_select = NUM_TEST_SAMPLES
        else:
            # 如果总样本数不足或等于5，则只取1个作为测试，其余作为训练
            print(f"警告：样本总数（{len(all_files)}）不足，仅抽取1个作为测试样本。")
            num_to_select = 1

        # 随机打乱列表
        random.shuffle(all_files)

        # 划分测试集和训练集
        test_files = all_files[:num_to_select]
        train_files = all_files[num_to_select:]

        print(f"划分结果: {len(train_files)} 个训练样本, {len(test_files)} 个测试样本。")

        # --- 5. 写入新的label-train和label-test文件 ---
        # 写入训练集标签
        with open(os.path.join(label_train_dir, category_filename), 'w', encoding='utf-8') as f:
            f.write('\n'.join(train_files))

        # 写入测试集标签
        with open(os.path.join(label_test_dir, category_filename), 'w', encoding='utf-8') as f:
            f.write('\n'.join(test_files))

        print("已生成新的训练和测试标签文件。")

        # --- 6. 移动测试集对应的图片文件 ---
        print("正在移动测试集图片...")
        for jpg_file in test_files:
            # 移动 orient 目录中的原始图
            src_orient_path = os.path.join(orient_dir, jpg_file)
            dest_orient_path = os.path.join(orient_test_dir, jpg_file)
            if os.path.exists(src_orient_path):
                shutil.move(src_orient_path, dest_orient_path)
            else:
                print(f"  警告: 图片 '{jpg_file}' 在目录 '{orient_dir}' 中未找到。")

            # 移动 transfer 目录中的风格迁移图
            src_transfer_path = os.path.join(transfer_dir, jpg_file)
            dest_transfer_path = os.path.join(transfer_test_dir, jpg_file)
            if os.path.exists(src_transfer_path):
                shutil.move(src_transfer_path, dest_transfer_path)
            else:
                print(f"  警告: 图片 '{jpg_file}' 在目录 '{transfer_dir}' 中未找到。")

    print("\n--- ✅ 所有处理已完成！ ---")


if __name__ == '__main__':
    # 假设您的 'label', 'orient', 'transfer' 目录与脚本在同一级
    # 如果不在，可以提供它们的父目录路径，例如 split_dataset('D:/my_project/')
    split_dataset()