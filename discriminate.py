import  torch
import os
from    torch.nn import functional as F
from    torch.utils.data import DataLoader
from    torchvision import transforms
from    torch import nn, optim
from PIL import Image
from torch.utils.data import  TensorDataset
from itertools import islice

'''判别器网络，充当生成器的对手，检查隐私标签在匿名化处理中的掩蔽程度'''

img_size = 144

transform = transforms.Normalize(mean=[0.500, 0.500, 0.500],
                                 std=[0.229, 0.224, 0.225])

class ResBlk(nn.Module):
    """
    resnet block
    """

    def __init__(self, ch_in, ch_out):
        """
        :param ch_in:
        :param ch_out:
        """
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
                nn.BatchNorm2d(ch_out)
            )
        self.Pool = nn.MaxPool2d(2, stride=2)


    def forward(self, x):
        """
        :param x: [b, ch, h, w]
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # short cut
        # extra module: [b, ch_in, h, w] => [b, ch_out, h, w]
        # element-wise add:
        out = self.extra(x) + out
        out = self.Pool(out)
        return out




class ResNet18(nn.Module):
# 使用5个独立的线性头，分别判断5个隐私标签的掩蔽程度

    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16)
        )
        # followed 4 blocks
        # [b, 64, h, w] => [b, 128, h ,w]
        self.blk1 = ResBlk(16, 16)
        # [b, 128, h, w] => [b, 256, h, w]
        self.blk2 = ResBlk(16, 32)
        # # [b, 256, h, w] => [b, 512, h, w]
        self.blk3 = ResBlk(32, 64)
        # # # [b, 512, h, w] => [b, 1024, h, w]
        self.blk4 = ResBlk(64, 128)
        self.lines = nn.ModuleList()
        for i in range(5):
            self.lines.append(nn.Sequential(
                nn.Linear(9 * 9 * 128, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
            ))

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))

        # [b, 64, h, w] => [b, 1024, h, w]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        # print(x.shape)
        x = x.view(x.size(0), -1)
        for i, line in enumerate(self.lines):
            y = line(x)
            y = torch.softmax(y, dim=1)
            if i == 0:
                a = y.unsqueeze(0)
            else:
                a = torch.cat([a, y.unsqueeze(0)], dim=0)
        return a.permute(1, 2, 0)

def load_img(img_path):
    img = Image.open(img_path).convert(
        'RGB')  # 使打开的图片通道为RGB格式,如果不使用.convert('RGB')进行转换的话，读出来的图像是RGBA四通道的，A通道为透明通道，该对深度学习模型训练来说暂时用不到，因此使用convert('RGB')进行通道转换。
    img = img.resize((img_size, img_size))  # 对图片进行裁剪，为512x512
    img = transforms.ToTensor()(img)
    img = transform(img).unsqueeze(0)  # unsqueeze升维，使数据格式符合[batch_size, n_channels, hight, width],[1,3,512,512]
    return img

def main():
    batchsz = 32
    transfer_path = './data/transfer/'
    transfer_img = None
    label_img = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for file in os.listdir(transfer_path):
        new_image = transfer_path + file
        if transfer_img is None:
            transfer_img = load_img(new_image).to(device)
        else:
            image_x = load_img(new_image).to(device)
            transfer_img = torch.cat([transfer_img, image_x], dim=0)
    transfer_img = TensorDataset(transfer_img)
    data_loader = torch.utils.data.DataLoader(transfer_img, batch_size=batchsz)

    result_dict = {}
    transfer_dict = {'800': 0, '1148': 1, '1817': 2, '2880': 3, '3899': 4, '4785': 5, '6067': 6, '6565': 7, '7769': 8,
                     '8692': 9, '9040': 10, '9295': 11}

    # 数据集文件夹路径，这里本python文件同级
    data_dir = "data"
    label_dir = os.path.join(data_dir, "label-train")
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
            # 从文件名中提取id值，这里根据新的文件名格式face_<具体id值>.txt进行提取
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
                        result_dict[img_name] = [transfer_dict[id_value], male_value, wavy_hair_value, oval_face_value,
                                                 pointy_nose_value, bags_under_eyes_value]

    for file in os.listdir(transfer_path):
        if label_img is None:
            label_img = torch.LongTensor([result_dict[file][1:]]).to(device)
        else:
            image_x = torch.LongTensor([result_dict[file][1:]]).to(device)
            label_img = torch.cat([label_img, image_x], dim=0)

    label_img = TensorDataset(label_img)
    label_loader = torch.utils.data.DataLoader(label_img, batch_size=batchsz)

    model = ResNet18()
    model.load_state_dict(torch.load('model_discriminate.pth'))
    model = model.to(device)

    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr= 3 * 1e-4)


    for epoch in range(100):

        model.train()
        for batchid, (x) in enumerate(data_loader):

            x = x[0].to(device)
            if batchid == 0:
                label = next(iter(label_loader))[0].to(device)
            else:
                label = list(islice(label_loader, batchid, batchid + 1))[0][0].to(device)

            logits = model(x)

            loss = criteon(logits, label)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch, 'loss:', loss.item())
    torch.save(model.state_dict(), 'model_discriminate.pth')

if __name__ == '__main__':
    main()