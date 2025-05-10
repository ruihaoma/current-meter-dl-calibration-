import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

class AmmeterDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        """
        参数:
            csv_file (str): CSV文件路径（如 data.csv）
            image_dir (str): 图片文件夹路径
            transform (callable, optional): 图像预处理变换
        """
        # 检查文件是否存在（增强错误提示）
        if not os.path.exists(csv_file):
            raise FileNotFoundError(
                f"CSV文件未找到: {csv_file}\n"
                f"当前工作目录: {os.getcwd()}"
            )
        if not os.path.exists(image_dir):
            raise NotADirectoryError(
                f"图片目录不存在: {image_dir}\n"
                f"目录内容: {os.listdir(os.path.dirname(image_dir))}"
            )

        # 读取CSV（增加编码格式处理）
        try:
            self.data = pd.read_csv(csv_file)
        except UnicodeDecodeError:
            self.data = pd.read_csv(csv_file, encoding='gbk')  # 尝试中文编码

        # 检查必要列是否存在
        if 'filename' not in self.data.columns:
            raise KeyError("CSV中必须包含 'filename' 列")
        if 'value' not in self.data.columns:
            raise KeyError("CSV中必须包含 'value' 列")

        self.image_dir = image_dir
        self.transform = transform

        # 预处理文件名（去除空格/换行符/引号）
        self.data['filename'] = self.data['filename'].astype(str).str.strip().str.strip('"\'')
        
        # 构建图片路径（支持多扩展名）
        self.image_paths = []
        missing_files = []
        for name in self.data['filename']:
            found = False
            for ext in ['.png', '.jpg', '.jpeg', '']:  # 尝试带扩展名和原始名
                path = os.path.join(self.image_dir, f"{name}{ext}")
                if os.path.exists(path):
                    self.image_paths.append(path)
                    found = True
                    break
            if not found:
                missing_files.append(name)

        if missing_files:
            raise FileNotFoundError(
                f"缺失 {len(missing_files)} 张图片，例如: {missing_files[:3]}...\n"
                f"搜索目录: {image_dir}\n"
                f"目录内容: {os.listdir(image_dir)[:5]}..."
            )

        self.labels = self.data['value'].tolist()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            with Image.open(image_path) as img:
                image = img.convert('RGB')
                if self.transform:
                    image = self.transform(image)
                label = torch.tensor([self.labels[idx]], dtype=torch.float32)
                return image, label
        except Exception as e:
            raise RuntimeError(
                f"加载图片失败: {image_path}\n"
                f"错误类型: {type(e).__name__}\n"
                f"详细信息: {str(e)}"
            )


def get_data_paths():
    """获取数据路径（根据你的实际情况定制）"""
    base_dir = os.path.join(os.path.expanduser("~"), "Desktop", "ammeter_project")
    
    # 你的实际CSV文件名是 data.csv
    csv_file = os.path.join(base_dir, "data.csv")
    
    # 假设图片在 images 文件夹
    image_dir = os.path.join(base_dir, "images")
    
    # 打印路径确认
    print(f"CSV文件路径: {csv_file}")
    print(f"图片目录路径: {image_dir}")
    
    return csv_file, image_dir


# 使用示例
if __name__ == "__main__":
    from torchvision import transforms

    # 1. 获取路径
    csv_file, image_dir = get_data_paths()

    # 2. 定义数据增强
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])

    # 3. 创建数据集
    try:
        dataset = AmmeterDataset(csv_file, image_dir, transform=transform)
        print(f"数据集加载成功，共 {len(dataset)} 张图片")
        image, label = dataset[0]
        print(f"示例图片尺寸: {image.shape}, 标签值: {label.item()}")
    except Exception as e:
        print(f"初始化数据集失败: {str(e)}")