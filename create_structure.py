import os

# 定义文件夹结构
structure = {
    "core": [],
    "services": [],
    "static": [],
    "templates": [],
    "models": [],
    "app.py": None  # 空文件
}

# 创建文件夹和文件
for item, subitems in structure.items():
    if subitems is None:  # 如果是文件
        open(item, "w").close()  # 创建空文件
    else:  # 如果是文件夹
        os.makedirs(item, exist_ok=True)
        for subitem in subitems:
            os.makedirs(os.path.join(item, subitem), exist_ok=True)

print("项目结构创建完成！")