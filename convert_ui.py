import os
import subprocess

# 源.ui文件目录
ui_dir = "ui"
# 目标Python文件目录
target_dir = "src/ui_forms"

# 创建目标目录
os.makedirs(target_dir, exist_ok=True)

# 转换所有.ui文件
for filename in os.listdir(ui_dir):
    if filename.endswith(".ui"):
        ui_path = os.path.join(ui_dir, filename)
        py_filename = f"ui_{filename[:-3]}.py"
        py_path = os.path.join(target_dir, py_filename)
        
        # 使用pyuic5转换
        subprocess.run([
            "pyuic5", 
            "-x", 
            ui_path, 
            "-o", 
            py_path
        ], check=True)
        print(f"转换完成: {py_path}")