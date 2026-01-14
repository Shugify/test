import os
import json

def generate_metadata(dataset_root, split_name, output_filename):
    """
    根据复杂的嵌套目录结构生成元数据文件。

    参数:
        dataset_root (str): 数据集的根目录，即包含 'train' 和 'test' 的目录。
        split_name (str): 要处理的数据集部分，'train' 或 'test'。
        output_filename (str): 输出的元数据文件名。
    """
    print(f"开始处理 '{split_name}' 数据集...")
    
    metadata_records = []
    split_path = os.path.join(dataset_root, split_name)

    # os.walk 会递归地遍历一个目录下的所有子目录和文件。
    # current_dir: 当前正在遍历的文件夹路径。
    # sub_dirs: 当前文件夹下的子文件夹列表。
    # files: 当前文件夹下的文件列表。
    for current_dir, sub_dirs, files in os.walk(split_path):
        # 遍历当前文件夹下的所有文件
        for filename in files:
            # 确保我们只处理图片文件，避免处理如 .DS_Store 等隐藏文件
            if not filename.lower().endswith(('.png', '.jpg')):
                continue

            # 获取从 split_path (例如 '.../idimage_v1/train') 到
            # 当前文件所在目录的相对路径。
            # 例如，如果 current_dir 是 '.../idimage_v1/train/aigc/biden'，
            # relative_path 会是 'aigc/biden'。
            relative_path = os.path.relpath(current_dir, split_path)

            # 如果图片直接在 train/ 或 test/ 目录下，没有分类文件夹，则跳过
            if relative_path == '.':
                continue

            # 我们需要的分类文件夹就是相对路径的第一部分。
            # 'aigc/biden'.split(os.sep) 会得到 ['aigc', 'biden']
            # os.sep 是系统自适应的路径分隔符 ('/' on Linux, '\\' on Windows)
            category_folder = relative_path.split(os.sep)[0]
            
            # 根据你的规则来分配标签
            # 如果分类文件夹的名字中包含 'real'，标签为0，否则为1
            label = 0 if 'real' in category_folder.lower() else 1

            # 构造图片的完整绝对路径
            image_full_path = os.path.join(current_dir, filename)

            # 创建一条元数据记录
            record = {
                "image_path": image_full_path,
                "label": label
            }
            metadata_records.append(record)

    # 将所有收集到的记录写入.jsonl文件
    with open(output_filename, 'w', encoding='utf-8') as f:
        for record in metadata_records:
            f.write(json.dumps(record) + '\n')

    print(f"'{split_name}' 的元数据文件已成功生成在: {output_filename}")
    print(f"共找到 {len(metadata_records)} 条记录。\n")

if __name__ == '__main__':
    dataset_root_path = "/remote-home/xujunhao/yodet/idimage_v1"
    jsonl_file_path="/root/sjj/test/test/new_dataset"
    
    # 为训练集生成元数据
    train_output_file = os.path.join(jsonl_file_path, "train_metadata.jsonl")
    generate_metadata(dataset_root_path, 'train', train_output_file)

    # 为测试集生成元数据
    test_output_file = os.path.join(jsonl_file_path, "val_metadata.jsonl")
    generate_metadata(dataset_root_path, 'test', test_output_file)