'''过滤主数据集的注释文件，创建每个类别的文本文件，记录此类下的所有图片'''

def filter_lines(input_file, output_file, target_suffix='2937'):
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    # 筛选出最后四位数字是2880的行
    filtered_lines = [line for line in lines if line.strip()[-4:] == target_suffix]

    # 将筛选结果写入新的文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.writelines(filtered_lines)

    print(f'筛选完成，结果已保存到 {output_file}')

# 使用示例
input_file = 'D:/BaiduNetdiskDownload/Large-scale CelebFaces Attributes (CelebA) Dataset/Large-scale CelebFaces Attributes (CelebA) Dataset/Anno/identity_CelebA.txt'  # 替换为你的输入文件路径
output_file = '../data2/label/face_2937.txt'  # 替换为你的输出文件路径

filter_lines(input_file, output_file)