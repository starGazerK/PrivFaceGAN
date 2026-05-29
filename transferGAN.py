import  torch
import os
from    torch.nn import functional as F
from    torch.utils.data import DataLoader
from    torchvision import transforms
from    torch import nn, optim
from PIL import Image
from torch.utils.data import  TensorDataset
from itertools import islice
import numpy as np

img_size = 144
batchsz = 32
channels = 3
lr = 0.0001
b1 = 0.5
b2 = 0.999
sample_interval = 400
img_shape = (channels, img_size, img_size)
image_path = './data/orient/'
transfer_path = './data/transfer/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Normalize(mean=[0.500, 0.500, 0.500],
                                 std=[0.229, 0.224, 0.225])

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128)
        )

        self.model = nn.Sequential(
            *block(9 * 9 * 128, 2048, normalize=False),
            *block(2048, 512),
            *block(512, 128),
            *block(128, 32),
            *block(32, 128),
            *block(128, 512),
            *block(512, 2048),
            nn.Linear(2048, 9 * 9 * 128),
            nn.Sigmoid()
        )

        self.deconv_layer = nn.Sequential(
            nn.ConvTranspose2d(128,64,3,2,1,1),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 3, 3, 2, 1, 1)
        )


    def forward(self, z):
        z = self.conv1(z)
        z = z.view(z.size(0), -1)
        img = self.model(z)
        img = img.view(img.size(0), 128, 9, 9)
        img = self.deconv_layer(img)
        return img

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
        # short cut.
        # extra module: [b, ch_in, h, w] => [b, ch_out, h, w]
        # element-wise add:
        out = self.extra(x) + out
        out = self.Pool(out)
        return out



class ResNet18_discriminator(nn.Module):

    def __init__(self):
        super(ResNet18_discriminator, self).__init__()

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

class ResNet18_recognization(nn.Module):

    def __init__(self):
        super(ResNet18_recognization, self).__init__()

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
        self.line = nn.Sequential(
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
        )

        self.outlayer = nn.Linear(64, 12)

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
        x = self.line(x)
        x = self.outlayer(x)
        x = torch.softmax(x, dim=1)

        return x

def load_img(img_path):
    img = Image.open(img_path).convert(
        'RGB')  # 使打开的图片通道为RGB格式,如果不使用.convert('RGB')进行转换的话，读出来的图像是RGBA四通道的，A通道为透明通道，该对深度学习模型训练来说暂时用不到，因此使用convert('RGB')进行通道转换。
    img = img.resize((img_size, img_size))  # 对图片进行裁剪，为512x512
    img = transforms.ToTensor()(img)
    img = transform(img).unsqueeze(0)  # unsqueeze升维，使数据格式符合[batch_size, n_channels, hight, width],[1,3,512,512]
    return img


result_dict = {}
result_dict_false = {}
transfer_dict = {'800': 0, '1148': 1, '1817': 2, '2880': 3, '3899': 4, '4785': 5, '6067': 6, '6565': 7, '7769': 8,
                 '8692': 9, '9040': 10, '9295': 11}

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

                    male_value_false = 1 if attr_dict[img_name]["Male"] == 0 else 1
                    wavy_hair_value_false = 1 if attr_dict[img_name]["Wavy_Hair"] == 0 else 1
                    oval_face_value_false = 1 if attr_dict[img_name]["Oval_Face"] == 0 else 1
                    pointy_nose_value_false = 1 if attr_dict[img_name]["Pointy_Nose"] == 0 else 1
                    bags_under_eyes_value_false = 1 if attr_dict[img_name]["Bags_Under_Eyes"] == 0 else 1
                    # 构建最终字典要求的格式
                    result_dict[img_name] = [transfer_dict[id_value], male_value, wavy_hair_value, oval_face_value,
                                             pointy_nose_value, bags_under_eyes_value]

                    result_dict_false[img_name] = [transfer_dict[id_value], male_value_false, wavy_hair_value_false, oval_face_value_false,
                                                   pointy_nose_value_false, bags_under_eyes_value_false]


content_img = None
transfer_img = None
discriminate_true_label = None
discriminate_false_label = None
recognization_label = None

for file in os.listdir(image_path):
    new_image = image_path + file
    if content_img is None:
        content_img = load_img(new_image).to(device)
    else:
        image_x = load_img(new_image).to(device)
        content_img = torch.cat([content_img, image_x], dim=0)

for file in os.listdir(transfer_path):
    new_image = transfer_path + file
    if transfer_img is None:
        transfer_img = load_img(new_image).to(device)
    else:
        image_x = load_img(new_image).to(device)
        transfer_img = torch.cat([transfer_img, image_x], dim=0)

for file in os.listdir(transfer_path):
    if discriminate_true_label is None:
        discriminate_true_label = torch.LongTensor([result_dict[file][1:]]).to(device)
    else:
        image_x = torch.LongTensor([result_dict[file][1:]]).to(device)
        discriminate_true_label = torch.cat([discriminate_true_label, image_x], dim=0)

for file in os.listdir(transfer_path):
    if discriminate_false_label is None:
        discriminate_false_label = torch.LongTensor([result_dict_false[file][1:]]).to(device)
    else:
        image_x = torch.LongTensor([result_dict_false[file][1:]]).to(device)
        discriminate_false_label = torch.cat([discriminate_false_label, image_x], dim=0)

for file in os.listdir(transfer_path):
    if recognization_label is None:
        recognization_label = torch.LongTensor([result_dict[file][0]]).to(device)
    else:
        image_x = torch.LongTensor([result_dict[file][0]]).to(device)
        recognization_label = torch.cat([recognization_label, image_x], dim=0)



content_img = TensorDataset(content_img)
content_loader = torch.utils.data.DataLoader(content_img, batch_size=batchsz)
transfer_img = TensorDataset(transfer_img)
transfer_loader = torch.utils.data.DataLoader(transfer_img, batch_size=batchsz)
discriminate_true_label = TensorDataset(discriminate_true_label)
discriminate_true_loader = torch.utils.data.DataLoader(discriminate_true_label, batch_size=batchsz)
discriminate_false_label = TensorDataset(discriminate_false_label)
discriminate_false_loader = torch.utils.data.DataLoader(discriminate_false_label, batch_size=batchsz)
recognization_label = TensorDataset(recognization_label)
recognization_loader = torch.utils.data.DataLoader(recognization_label, batch_size=batchsz)

generator = Generator()
generator.load_state_dict(torch.load('model_generator.pth'))
generator.to(device)
discriminate = ResNet18_discriminator()
discriminate.load_state_dict(torch.load('model_discriminate.pth'))
discriminate = discriminate.to(device)
recognization = ResNet18_recognization()
recognization.load_state_dict(torch.load('model_recognization.pth'))
recognization = recognization.to(device)

criteon_discriminate = nn.CrossEntropyLoss().to(device)
criteon_recognization = nn.CrossEntropyLoss().to(device)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_R = optim.Adam(recognization.parameters(), lr= 3 * 1e-4)
optimizer_D = optim.Adam(discriminate.parameters(), lr= 1e-4)

n_epochs = 13# 逐步改小，直到测试集准确率达到要求，趋向于100%

for epoch in range(n_epochs):
    for batchid, (x) in enumerate(content_loader):
        x = x[0].to(device)
        if batchid == 0:
            y = next(iter(transfer_loader))[0].to(device)
        else:
            y = list(islice(transfer_loader, batchid, batchid + 1))[0][0].to(device)
        if batchid == 0:
            discriminate_false_label = next(iter(discriminate_false_loader))[0].to(device)
        else:
            discriminate_false_label = list(islice(discriminate_false_loader, batchid, batchid + 1))[0][0].to(device)
        if batchid == 0:
            discriminate_true_label = next(iter(discriminate_true_loader))[0].to(device)
        else:
            discriminate_true_label = list(islice(discriminate_true_loader, batchid, batchid + 1))[0][0].to(device)
        if batchid == 0:
            recognization_label = next(iter(recognization_loader))[0].to(device)
        else:
            recognization_label = list(islice(recognization_loader, batchid, batchid + 1))[0][0].to(device)
        x = generator(x)
        optimizer_G.zero_grad()

        g_loss = criteon_discriminate(discriminate(x), discriminate_false_label) + criteon_recognization(recognization(x), recognization_label)
        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        real_loss = criteon_discriminate(discriminate(y), discriminate_true_label)
        fake_loss = criteon_discriminate(discriminate(x.detach()), discriminate_true_label)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        optimizer_R.zero_grad()
        recognization_loss = criteon_recognization(recognization(x.detach()), recognization_label)
        recognization_loss.backward()
        optimizer_R.step()

        print("[Epoch %d/%d] [Batch %d/%d] [D Loss: %f] [G Loss: %f] [R Loss: %f]" % (
            epoch, n_epochs, batchid, len(content_loader), d_loss.item(), g_loss.item(), recognization_loss.item()))

torch.save(generator.state_dict(), 'model_generator.pth')
torch.save(recognization.state_dict(), 'model_recognization.pth')
torch.save(discriminate.state_dict(), 'model_discriminate.pth')

