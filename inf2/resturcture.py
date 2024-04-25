import os
import shutil

def process_audio_folders(root_dir):
    # 遍历根目录下的所有子目录
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.wav'):
                # 构建原始文件的完整路径
                original_path = os.path.join(subdir, file)
                # 获取子文件夹的名称
                folder_name = os.path.basename(subdir)
                # 新的文件名是子文件夹的名称
                new_file_name = folder_name + '.wav'
                # 新文件的路径是根目录下
                new_file_path = os.path.join(root_dir, new_file_name)
                # 移动文件
                shutil.move(original_path, new_file_path)
                # 检查子目录是否为空，如果为空则删除
                if not os.listdir(subdir):
                    os.rmdir(subdir)
                print(f"Processed {new_file_name}")

if __name__ == '__main__':
    # 指定要处理的根目录路径
    #root_directory = input("Enter the path to the directory: ")
    process_audio_folders("/home/weizhenbian/mycode/inf2/out")
