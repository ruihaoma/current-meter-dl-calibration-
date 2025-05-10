import matplotlib.pyplot as plt
import numpy as np
import os
import csv


def draw_ammeter(value, save_path="output.png"):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis("off")

    # 画圆形表盘
    circle = plt.Circle((0, 0), 1.0, fill=False, linewidth=3)
    ax.add_patch(circle)

    # 画刻度线和刻度数值（0 ~ 10）
    for i in range(11):
        angle = np.pi * (1 - i / 10)  # 180° to 0°
        x = np.cos(angle)
        y = np.sin(angle)
        ax.plot([0.9*x, x], [0.9*y, y], color='black', linewidth=2)
        ax.text(0.8*x, 0.8*y, str(i), ha='center', va='center', fontsize=10)

    # 画指针（以电流值决定角度）
    pointer_angle = np.pi * (1 - value / 10)  # 映射到 180° - 0°
    px, py = 0.8 * np.cos(pointer_angle), 0.8 * np.sin(pointer_angle)
    ax.plot([0, px], [0, py], color='red', linewidth=3)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


# 创建保存图片的目录
image_dir = "images"
os.makedirs(image_dir, exist_ok=True)

# 创建 CSV 文件并写入表头
with open("data.csv", mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["filename", "value"])

    # 生成 100 张图片并写入对应标签
    for i in range(100):
        value = np.random.uniform(0, 10)  # 随机电流值（模拟指针位置）
        filename = f"ammeter_{i}.png"
        filepath = os.path.join(image_dir, filename)
        draw_ammeter(value, save_path=filepath)
        writer.writerow([filepath, value])
        print(f"生成: {filepath}，值: {value:.2f} A")