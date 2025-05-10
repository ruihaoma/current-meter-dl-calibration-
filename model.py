import torch
import torch.nn as nn
import torch.nn.functional as F

class AmmeterModel(nn.Module):
    def __init__(self):
        super(AmmeterModel, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # 定义全连接层
        self.fc1 = nn.Linear(64 * 16 * 16, 512)  # 假设输入图像大小为128x128
        self.fc2 = nn.Linear(512, 1)  # 输出一个连续值（回归任务）

    def forward(self, x):
        # 卷积层 + 激活函数 + 池化层
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # 展平操作
        x = x.view(-1, 64 * 16 * 16)  # 展平为一维向量

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

