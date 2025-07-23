import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import wandb # 导入wandb

# ---------- 路径配置 ----------
data_dir = 'dataset'
model_dir = 'model'
os.makedirs(model_dir, exist_ok=True)

# ---------- 使用GPU ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------- 定义超参数 (方便wandb记录) ----------
config = {
    "learning_rate": 1e-4,
    "batch_size": 4,
    "num_epochs": 10,
    "architecture": "ResNet-18"
}

# ---------- wandb 初始化 ----------
# 1. 开始一个新的 wandb run 来跟踪这个实验
wandb.init(
    project="pytorch-catdog-classification", # 在wandb中显示的项目名称
    config=config # 传入超参数字典
)

# ---------- 数据预处理 ----------
# 注意：为了能正确显示图片，我们从[0.5, 0.5, 0.5]反归一化回[0, 1]
def denormalize(tensor):
    return tensor * 0.5 + 0.5

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ]),
}

# ---------- 加载数据 ----------
batch_size = wandb.config.batch_size # 从wandb.config中获取batch_size
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ['train', 'val']
}
dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True)
    for x in ['train', 'val']
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print("Classes:", class_names)

# ---------- 构建模型 ----------
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)

# ---------- 损失和优化器 ----------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate) # 从wandb.config获取学习率

# ---------- 训练函数 ----------
def train_model(model, criterion, optimizer, num_epochs=10):
    # 告诉wandb开始监控模型
    wandb.watch(model, criterion, log="all", log_freq=10)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss, running_corrects = 0.0, 0

            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()

                # 在每个epoch的第一个batch记录训练图片
                if phase == 'train' and i == 0:
                    # 反归一化并转换为wandb.Image格式
                    log_images = [wandb.Image(denormalize(img.cpu()), caption=f"Pred: {class_names[pred]}, Truth: {class_names[truth]}")
                                  for img, pred, truth in zip(inputs, preds, labels)]
                    # 记录图片到wandb
                    wandb.log({"Training Images": log_images}, step=epoch)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

            # 使用 wandb.log() 记录 loss 和 accuracy
            # 我们将训练和验证指标分开记录，方便在图表中对比
            log_data = {
                f"{phase}_loss": epoch_loss,
                f"{phase}_accuracy": epoch_acc
            }
            # 使用epoch作为x轴
            wandb.log(log_data, step=epoch)

    return model

# ---------- 启动训练 ----------
model = train_model(model, criterion, optimizer, num_epochs=wandb.config.num_epochs)

# ---------- 保存模型 ----------
model_path = os.path.join(model_dir, "resnet18_catdog.pth")
torch.save(model.state_dict(), model_path)
print("Model saved to:", model_path)

# (可选) 标记run结束
wandb.finish()