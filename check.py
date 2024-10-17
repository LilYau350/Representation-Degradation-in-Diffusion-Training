import os
from PIL import Image, PngImagePlugin

Image.MAX_IMAGE_PIXELS = None
PngImagePlugin.MAX_TEXT_CHUNK = 1024 * (2 ** 20)  # 1024MB
PngImagePlugin.MAX_TEXT_MEMORY = 128 * (2 ** 20)  # 128MB

def check_images_in_directory(directory):
    problem_images = []  # 存储有问题图像的列表
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)
                img.verify()  # 验证图像文件的完整性
            except (IOError, SyntaxError, ValueError) as e:
                print(f"问题图像: {file_path}, 错误: {e}")
                problem_images.append(file_path)  # 记录问题图像
                continue  # 遇到问题继续处理下一个文件
    return problem_images

if __name__ == "__main__":
    # 替换为你的图像文件夹路径
    problem_images = check_images_in_directory("D:/迅雷下载/ImageNet_256x256")
    
    # 保存问题图像路径到文件中
    with open("problem_images.txt", "a") as f:  # 使用 'a' 模式追加保存
        for image_path in problem_images:
            f.write(f"{image_path}\n")
