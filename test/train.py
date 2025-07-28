import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import wandb
import json
from PIL import Image
import copy
from tqdm import tqdm
import time # 【计时】导入 time 模块

# ---------- 路径配置 ----------
# ... (这部分不变)
data_dir = 'new_dataset'
model_dir = 'model'
os.makedirs(model_dir, exist_ok=True)


# --------- 定义自己的Dataset 和 collate_fn ---------
# ... (这部分不变)
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

def denormalize(tensor):
    return tensor * 0.5 + 0.5

# ---------- 训练函数 ----------
def train_model(model, criterion, optimizer, device, dataloaders, dataset_sizes, class_names, num_epochs=10):
    wandb.watch(model, criterion, log="all", log_freq=10)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        # 【计时】记录整个 epoch 的墙上时钟时间
        epoch_start_time = time.time()

        for phase in ['train', 'val']:
            # 【计时】为每个阶段内的各个部分耗时创建一个字典来累加
            phase_timings = {
                "data_loading_time": 0.0,
                "to_device_time": 0.0,
                "forward_time": 0.0,
                "backward_time": 0.0,
                "optimizer_step_time": 0.0
            }
            
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss, running_corrects, num_samples = 0.0, 0, 0
            
            # 【计时】记录上一个批次结束的时间点，用于计算数据加载时间
            batch_end_time = time.time()

            progress_bar = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch+1}")
            for i, (inputs, labels) in enumerate(progress_bar):
                # 【计时】数据加载/迭代时间 = 当前时间 - 上一批次处理结束的时间
                phase_timings["data_loading_time"] += time.time() - batch_end_time
                
                if inputs is None:
                    batch_end_time = time.time() # 更新时间以便下一次计算
                    continue
                
                # --- GPU 操作计时开始 ---
                if device.type == 'cuda':
                    # 使用 torch.cuda.Event 进行精确的 GPU 计时
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    
                    # 计时: 数据从 CPU -> GPU
                    torch.cuda.synchronize() # 保证之前的操作已完成
                    start_event.record()
                    inputs, labels = inputs.to(device), labels.to(device)
                    end_event.record()
                    torch.cuda.synchronize()
                    phase_timings["to_device_time"] += start_event.elapsed_time(end_event) / 1000.0 # 转换为秒

                else: # CPU 上的计时
                    to_device_start = time.time()
                    inputs, labels = inputs.to(device), labels.to(device)
                    phase_timings["to_device_time"] += time.time() - to_device_start

                num_samples += inputs.size(0)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    # 计时: 模型前向传播
                    if device.type == 'cuda':
                        start_event.record()
                    else:
                        forward_start = time.time()

                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if device.type == 'cuda':
                        end_event.record()
                        torch.cuda.synchronize()
                        phase_timings["forward_time"] += start_event.elapsed_time(end_event) / 1000.0
                    else:
                        phase_timings["forward_time"] += time.time() - forward_start

                    if phase == 'train':
                        # 计时: 模型反向传播
                        if device.type == 'cuda':
                            start_event.record()
                        else:
                            backward_start = time.time()
                        
                        loss.backward()

                        if device.type == 'cuda':
                            end_event.record()
                            torch.cuda.synchronize()
                            phase_timings["backward_time"] += start_event.elapsed_time(end_event) / 1000.0
                        else:
                            phase_timings["backward_time"] += time.time() - backward_start

                        # 计时: 优化器更新
                        if device.type == 'cuda':
                            start_event.record()
                        else:
                            optimizer_start = time.time()

                        optimizer.step()

                        if device.type == 'cuda':
                            end_event.record()
                            torch.cuda.synchronize()
                            phase_timings["optimizer_step_time"] += start_event.elapsed_time(end_event) / 1000.0
                        else:
                            phase_timings["optimizer_step_time"] += time.time() - optimizer_start


                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # 【计时】更新批次结束时间点
                batch_end_time = time.time()


            if num_samples == 0: continue

            epoch_loss = running_loss / num_samples
            epoch_acc = running_corrects.double() / num_samples
            
            # 【计时】打印每个阶段的详细耗时
            print(f"\n--- {phase.capitalize()} Phase Timing Breakdown (Epoch {epoch+1}) ---")
            total_phase_time = sum(phase_timings.values())
            for key, value in phase_timings.items():
                print(f"{key:<22}: {value:.4f}s ({(value/total_phase_time)*100:.2f}%)")
            print("-------------------------------------------------")

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                model_path = os.path.join(model_dir, "resnet18_best.pth")
                torch.save(model.state_dict(), model_path)
                print(f"新最佳模型已保存至 {model_path}，准确率: {best_acc:.4f}")

            log_data = {f"{phase}_loss": epoch_loss, f"{phase}_accuracy": epoch_acc}
            # 【计时】将详细耗时记录到 wandb
            for key, value in phase_timings.items():
                log_data[f"timing/{phase}_{key}"] = value
            wandb.log(log_data, step=epoch)
        
        # 【计时】在 epoch 结束后计算并打印总耗时
        epoch_time_elapsed = time.time() - epoch_start_time
        print(f"\nEpoch {epoch+1} Wall-Time: {epoch_time_elapsed // 60:.0f}分 {epoch_time_elapsed % 60:.0f}秒")
        wandb.log({"timing/epoch_total_wall_time": epoch_time_elapsed}, step=epoch)


    print(f'\n最佳验证准确率 (Best val Acc): {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model

def main():
    main_start_time = time.time() # 【计时】脚本总开始时间
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    config = {"learning_rate": 1e-4, "batch_size": 256, "num_epochs": 10, "architecture": "ResNet-18"}
    wandb.init(project="pytorch-250725classification-profiled", config=config) # 新开一个项目

    data_transforms = {
        'train': transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)]),
        'val': transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)]),
    }

    # 【计时】数据加载设置部分的耗时
    data_setup_start = time.time()
    train_filename = "train_metadata.jsonl"
    val_filename = "val_metadata.jsonl"
    image_datasets = {
        'train': MyDataset(os.path.join(data_dir, train_filename), data_transforms['train']),
        'val': MyDataset(os.path.join(data_dir, val_filename), data_transforms['val'])
    }
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=wandb.config.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn_safe, pin_memory=True),
        'val': DataLoader(image_datasets['val'], batch_size=wandb.config.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn_safe, pin_memory=True)
    }
    print(f"--- Data setup took: {time.time() - data_setup_start:.4f}s ---")

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = ['True_Image', 'False_Image'] 
    
    # 【计时】模型设置部分的耗时
    model_setup_start = time.time()
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) # 使用新API
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(device)
    print(f"--- Model setup took: {time.time() - model_setup_start:.4f}s ---")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

    # 启动训练
    model = train_model(model, criterion, optimizer, device, dataloaders, dataset_sizes, class_names, num_epochs=wandb.config.num_epochs)

    # 保存最终的最佳模型
    model_path = os.path.join(model_dir, "resnet18_final_best.pth")
    torch.save(model.state_dict(), model_path)
    print("Final best model saved to:", model_path)

    wandb.finish()
    print(f"\n--- Total script execution time: {time.time() - main_start_time:.2f}s ---")

if __name__ == '__main__':
    main()