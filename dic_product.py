import os

# 用于存储最终的字典结果
result_dict = {}
transfer_dict = {'800':0,'1148':1,'1817':2,'2880':3,'3899':4,'4785':5,'6067':6,'6565':7,'7769':8,'8692':9,'9040':10,'9295':11}

# 数据集文件夹路径，这里本python文件同级
data_dir = "data"
label_dir = os.path.join(data_dir, "label")
list_attr_file = "list_attr_celeba.txt"

# 读取list_attr_celeba.txt文件内容，构建以图片名为键，属性值列表为值的字典
attr_dict = {}
with open(list_attr_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    attr_names = lines[1].strip().split()
    for line in lines[2:]:
        parts = line.strip().split()
        img_name = parts[0]
        attr_values = [int(x) for x in parts[1:]]
        attr_dict[img_name] = dict(zip(attr_names, attr_values))

# 遍历12个face_*.txt文件
for face_id_file in os.listdir(label_dir):
    if face_id_file.startswith("face_") and face_id_file.endswith(".txt"):
        # 从文件名中提取 id 值，这里根据新的文件名格式face_<具体id值>.txt进行提取
        id_value = face_id_file[5:-4]
        face_id_path = os.path.join(label_dir, face_id_file)
        with open(face_id_path, 'r', encoding='utf-8') as f:
            for img_name in f.readlines():
                img_name = img_name.strip()
                if img_name in attr_dict:
                    
                    # 获取对应图片的五个属性值
                    male_value = 1 if attr_dict[img_name]["Male"] == 1 else 0
                    wavy_hair_value = 1 if attr_dict[img_name]["Wavy_Hair"] == 1 else 0
                    oval_face_value = 1 if attr_dict[img_name]["Oval_Face"] == 1 else 0
                    pointy_nose_value = 1 if attr_dict[img_name]["Pointy_Nose"] == 1 else 0
                    bags_under_eyes_value = 1 if attr_dict[img_name]["Bags_Under_Eyes"] == 1 else 0
                    # 构建最终字典要求的格式
                    result_dict[img_name] = [transfer_dict[id_value], male_value, wavy_hair_value, oval_face_value, pointy_nose_value, bags_under_eyes_value]
