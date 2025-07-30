# train.py (Refactored)

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import wandb
import yaml
import argparse
import shutil
import json
from PIL import Image
import copy
from tqdm import tqdm
import time
from transformers import ViTForImageClassification


# ---------- 数据集定义 (这部分不变) ----------
class MyDataset(Dataset):
    def __init__(self, metadata_file: str, transform: transforms.Compose = None):
        if not os.path.exists(metadata_file):
             raise FileNotFoundError(f"元数据文件未找到：{metadata_file}")
        self.metadata_file=metadata_file
        self.transform=transform
        self.data = self._load_metadata()

    def _load_metadata(self)-> list:
        print(f"正在从 {self.metadata_file} 加载元数据...")
        metadata_list = []
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    metadata_list.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    print(f"警告：跳过格式错误的行: {line.strip()}")
        print(f"元数据加载完毕，共 {len(metadata_list)} 条记录。")
        return metadata_list
    
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        record = self.data[idx]
        image_path = record.get("image_path")
        label = record.get("label")
        if image_path is None or label is None:
            raise KeyError(f"记录 {record} 中缺少 'image_path' 或 'label' 键。")
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            return None, None
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label

def collate_fn_safe(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0:
        return (None, None) 
    return torch.utils.data.dataloader.default_collate(batch)


# ---------- 训练函数 (修改了模型保存路径) ----------
def train_model(model, criterion, optimizer, device, dataloaders, num_epochs, output_dir):
    wandb.watch(model, criterion, log="all", log_freq=100)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        epoch_start_time = time.time()

        for phase in ['train', 'val']:
            phase_timings = { "data_loading_time": 0.0, "to_device_time": 0.0, "forward_time": 0.0, "backward_time": 0.0, "optimizer_step_time": 0.0 }
            if phase == 'train': model.train()
            else: model.eval()

            running_loss, running_corrects, num_samples = 0.0, 0, 0
            batch_end_time = time.time()
            progress_bar = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch+1}")
            
            for inputs, labels in progress_bar:
                phase_timings["data_loading_time"] += time.time() - batch_end_time
                if inputs is None:
                    batch_end_time = time.time()
                    continue

                to_device_start = time.time()
                inputs, labels = inputs.to(device), labels.to(device)
                phase_timings["to_device_time"] += time.time() - to_device_start

                num_samples += inputs.size(0)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    forward_start = time.time()
                    outputs = model(inputs)
                    logits=outputs.logits
                    _, preds = torch.max(logits, 1)
                    loss = criterion(logits, labels)
                    phase_timings["forward_time"] += time.time() - forward_start

                    if phase == 'train':
                        backward_start = time.time()
                        loss.backward()
                        phase_timings["backward_time"] += time.time() - backward_start
                        
                        optimizer_start = time.time()
                        optimizer.step()
                        phase_timings["optimizer_step_time"] += time.time() - optimizer_start

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                batch_end_time = time.time()

            if num_samples == 0: continue
            epoch_loss = running_loss / num_samples
            epoch_acc = running_corrects.double() / num_samples
            
            print(f"\n--- {phase.capitalize()} Phase Timing Breakdown (Epoch {epoch+1}) ---")
            total_phase_time = sum(phase_timings.values())
            if total_phase_time > 0:
                for key, value in phase_timings.items():
                    print(f"{key:<22}: {value:.4f}s ({(value/total_phase_time)*100:.2f}%)")
            
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")
            
            # **修改点**: 保存模型到配置好的输出目录
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                model_path = os.path.join(output_dir, "best_model.pth")
                torch.save(model.state_dict(), model_path)
                print(f"新最佳模型已保存至 {model_path}，准确率: {best_acc:.4f}")

            log_data = {f"{phase}_loss": epoch_loss, f"{phase}_accuracy": epoch_acc}
            for key, value in phase_timings.items():
                log_data[f"timing/{phase}_{key}"] = value
            wandb.log(log_data, step=epoch)
        
        epoch_time_elapsed = time.time() - epoch_start_time
        print(f"\nEpoch {epoch+1} Wall-Time: {epoch_time_elapsed // 60:.0f}分 {epoch_time_elapsed % 60:.0f}秒")
        wandb.log({"timing/epoch_total_wall_time": epoch_time_elapsed}, step=epoch)

    print(f'\n最佳验证准确率 (Best val Acc): {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model

# ---------- 主函数 (核心修改区域) ----------
def main():
    # 1. 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="从YAML配置文件运行可复现的训练流程。")
    parser.add_argument('--config', type=str, required=True, help='指向 config.yaml 文件的路径')
    parser.add_argument('--shell_path', type=str, required=True, help='启动此脚本的 .sh 文件的路径')
    args = parser.parse_args()

    # 2. 加载并解析 YAML 配置文件
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print("配置加载成功!")
    
    # 3. 创建唯一的实验输出目录
    output_dir = os.path.join(config['experiment']['output_dir'], config['experiment']['name'])
    os.makedirs(output_dir, exist_ok=True)
    print(f" 所有输出将保存至: {output_dir}")

    # 4. [核心] 备份配置文件和启动脚本，保证可复现性
    shutil.copy(args.config, os.path.join(output_dir, 'config.yaml'))
    shutil.copy(args.shell_path, os.path.join(output_dir, os.path.basename(args.shell_path)))
    print(" 配置文件和启动脚本已备份。")

    # 5. 初始化 Weights & Biases
    wandb.init(
        project=config['wandb']['project'],
        entity=config['wandb']['entity'],
        name=config['experiment']['name'],
        config=config # 将所有配置上传到WandB
    )
    print(f" WandB 初始化成功! 访问 {wandb.run.url} 查看实时面板。")

    # 6. 设置设备
    if config['training']['device'] == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config['training']['device'])
    print(f"使用设备: {device}")

    # 7. 准备数据
    data_transforms = {
        'train': transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)]),
        'val': transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)]),
    }
    image_datasets = {
        'train': MyDataset(os.path.join(config['data']['base_dir'], config['data']['train_metadata']), data_transforms['train']),
        'val': MyDataset(os.path.join(config['data']['base_dir'], config['data']['val_metadata']), data_transforms['val'])
    }
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=config['training']['batch_size'], shuffle=(x=='train'), 
                      num_workers=config['data']['num_workers'], collate_fn=collate_fn_safe, pin_memory=True)
        for x in ['train', 'val']
    }

    # 8. 构建模型
    model = ViTForImageClassification.from_pretrained(
        config['model']['architecture'],
        num_labels=config['model']['num_classes'],     
        ignore_mismatched_sizes=True
    )
    model = model.to(device)

    # 9. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # 10. 启动训练
    main_start_time = time.time()
    model = train_model(model, criterion, optimizer, device, dataloaders, 
                        num_epochs=config['training']['epochs'],
                        output_dir=output_dir)

    # 11. 保存最终模型并结束
    final_model_path = os.path.join(output_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"最终模型已保存至: {final_model_path}")
    
    wandb.finish()
    print(f"\n--- 脚本总执行时间: {(time.time() - main_start_time)/60:.2f} 分钟 ---")


if __name__ == '__main__':
    main()