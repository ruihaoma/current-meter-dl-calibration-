import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from dataset import AmmeterDataset
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import os

# 超参数
num_epochs = 30
batch_size = 16
learning_rate = 0.0005

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
])

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 绘图函数：仅显示指针和标签框
def draw_pointer_on_ammeter(image_path, prediction):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    pointer_angle = np.pi * (1 - prediction / 10)
    pointer_length = 0.4 * min(width, height)
    pointer_x = width / 2 + pointer_length * np.cos(pointer_angle)
    pointer_y = height / 2 - pointer_length * np.sin(pointer_angle)

    draw = ImageDraw.Draw(image)
    draw.line([width / 2, height / 2, pointer_x, pointer_y], fill="red", width=4)

    # 标签框位置
    tag_w, tag_h = 80, 30
    tag_x = pointer_x + 10
    tag_y = pointer_y - tag_h / 2
    draw.rectangle([(tag_x, tag_y), (tag_x + tag_w, tag_y + tag_h)], fill="black")
    draw.text((tag_x + 5, tag_y + 5), f"{prediction:.2f} A", fill="white")

    plt.imshow(image)
    plt.axis("off")
    plt.show()

# 预测函数
def predict_image(image_path, model, transform):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(input_tensor).item()

    print(f"预测结果：{prediction:.2f} A")
    draw_pointer_on_ammeter(image_path, prediction)

# 主程序入口
def main():
    # 数据路径
    csv_path = 'data.csv'
    image_dir = "C:\Users\14726\Desktop\ammeter_project\images"
    dataset = AmmeterDataset(csv_file=csv_path, image_dir=image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 加载预训练模型
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 模型训练
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.view(-1, 1).to(device).float()

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

    print("Training finished.")
    torch.save(model.state_dict(), 'ammeter_resnet18.pth')
    print("Model saved successfully.")

    # 测试图片预测'
    test_image_path = "C:\Users\14726\Desktop\ammeter_project\images"
    predict_image(test_image_path, model, transform)

# 安全运行主函数（避免 Windows multiprocessing 错误）
if __name__ == '__main__':
    main()
