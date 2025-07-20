import os # 导入 'os' 模块，用于与操作系统交互，比如创建目录或拼接文件路径。
import torch # 导入 PyTorch 的核心库，这是所有张量操作和神经网络构建的基础。
import torch.nn as nn # 导入 PyTorch 的 'nn' 模块，它包含了所有神经网络的构建块（例如，全连接层 Linear, 卷积层 Conv2d）。
import torch.optim as optim # 导入 'optim' 模块，它提供了各种优化算法（如 SGD, Adam）来更新模型的权重。
from torchvision import datasets, models, transforms # 从 torchvision 库中导入特定的模块：
# 'datasets' 用于常用的数据集（例如 ImageFolder，可以方便地加载按文件夹组织的图像数据）。
# 'models' 用于预训练的模型（例如 ResNet）。
# 'transforms' 用于图像的各种预处理和增强操作。
from torch.utils.data import DataLoader # 导入 DataLoader，它能高效地以批次（batch）形式加载数据。

# ---------- 路径配置 ---------- # 这是一个注释，表示以下是关于文件路径的配置。
data_dir = 'dataset' # 定义你的图像数据集所在的根目录（例如，里面会有 'train' 和 'val' 子文件夹）。
model_dir = 'model' # 定义训练好的模型将要保存的目录。
os.makedirs(model_dir, exist_ok=True) # 如果 'model' 目录不存在，则创建它。'exist_ok=True' 参数表示如果目录已经存在，则不会引发错误。

# ---------- 使用GPU ---------- # 注释，表示以下是关于设备（GPU/CPU）的设置。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 检查是否可用 CUDA（NVIDIA GPU 的并行计算平台）。如果可用，则使用 'cuda' 设备；否则，退回到 'cpu'（中央处理器）。
print("Using device:", device) # 打印出当前将用于计算的设备是 GPU 还是 CPU。

# ---------- 数据预处理 ---------- # 注释，表示以下是数据预处理的设置。
data_transforms = { # 定义一个字典，包含训练集和验证集不同的数据转换（预处理）规则。
    'train': transforms.Compose([ # 对训练数据应用一系列转换操作：
        transforms.Resize((224, 224)), # 将图像大小统一调整为 224x224 像素。这是许多预训练模型（如 ResNet）的标准输入尺寸。
        transforms.RandomHorizontalFlip(), # 以 0.5 的概率随机对图像进行水平翻转。这是一种数据增强技术，可以增加数据的多样性，帮助模型更好地泛化。
        transforms.ToTensor(), # 将 PIL Image 或 NumPy ndarray 格式的图像转换为 PyTorch 张量 (Tensor)。它还会将像素值从 [0, 255] 归一化到 [0.0, 1.0]。
        transforms.Normalize([0.5]*3, [0.5]*3) # 对图像张量进行标准化。它会将每个颜色通道的像素值调整为均值为 0.5，标准差为 0.5。这意味着像素值将被缩放到 [-1, 1] 之间。这有助于模型的训练收敛。
    ]),
    'val': transforms.Compose([ # 对验证数据应用一系列转换操作（通常比训练集少，因为验证集不需要数据增强）：
        transforms.Resize((224, 224)), # 将图像大小统一调整为 224x224 像素。
        transforms.ToTensor(), # 将图像转换为 PyTorch 张量。
        transforms.Normalize([0.5]*3, [0.5]*3) # 对图像张量进行标准化。
    ]),
}

# ---------- 加载数据 ---------- # 注释，表示以下是数据加载的设置。
batch_size = 4 # 定义每个训练或验证批次中包含的图像数量。较大的批次大小通常能更稳定地估计梯度，但可能需要更多内存。
image_datasets = { # 创建一个字典，用于存储训练集和验证集的图像数据集。
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) # 对于 'train' 和 'val' 键：
    # `os.path.join(data_dir, x)` 会拼接出完整的数据集路径，例如 'dataset/train' 和 'dataset/val'。
    # `datasets.ImageFolder` 会自动识别这些路径下的子文件夹作为类别，并加载其中的图像。
    # `data_transforms[x]` 会应用对应的数据转换规则。
    for x in ['train', 'val'] # 这是一个字典推导式，循环遍历 'train' 和 'val'。
}
dataloaders = { # 创建一个字典，用于存储训练集和验证集的 DataLoader。
    x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) # 对于 'train' 和 'val' 键：
    # `DataLoader` 接收一个数据集 (`image_datasets[x]`)，并根据 `batch_size` 和 `shuffle` 参数创建可迭代的数据加载器。
    # `shuffle=True` (通常用于训练集) 表示在每个 epoch 开始时打乱数据，以确保模型不会学习到数据的固定顺序。
    for x in ['train', 'val']
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']} # 创建一个字典，记录训练集和验证集中图像的总数量。
class_names = image_datasets['train'].classes # 从训练数据集中获取所有类别的名称。`ImageFolder` 会根据子文件夹名称自动识别。
print("Classes:", class_names) # 打印出识别到的类别名称。

# ---------- 构建模型 ---------- # 注释，表示以下是模型构建的部分。
model = models.resnet18(pretrained=True) # 加载一个预训练的 ResNet18 模型。`pretrained=True` 表示下载并在 ImageNet 数据集上预训练好的权重。使用预训练模型可以显著加速训练，尤其是在数据量不足时。
model.fc = nn.Linear(model.fc.in_features, 2) # 修改 ResNet18 的最后一层（全连接层，通常命名为 `fc`）。
# `model.fc.in_features` 获取原始全连接层的输入特征数量。
# `nn.Linear(input_features, output_features)` 创建一个新的全连接层。
# 这里 `2` 表示输出类别数量，因为这是一个二分类任务（例如，猫和狗）。这样，模型就能适应你的特定任务。
model = model.to(device) # 将模型移动到之前确定的设备上（GPU 或 CPU）。如果使用 GPU，这将允许模型在 GPU 上进行计算。

# ---------- 损失和优化器 ---------- # 注释，表示以下是损失函数和优化器的设置。
criterion = nn.CrossEntropyLoss() # 定义损失函数。`nn.CrossEntropyLoss` 适用于多分类任务（包括二分类），它结合了 LogSoftmax 和 NLLLoss，常用于分类问题。
optimizer = optim.Adam(model.parameters(), lr=1e-4) # 定义优化器。`optim.Adam` 是一种常用的优化算法，通常在实践中表现良好。
# `model.parameters()` 告诉优化器要更新哪些参数（即模型的所有可学习权重和偏置）。
# `lr=1e-4` 设置学习率（learning rate），控制每次参数更新的步长大小。

# ---------- 训练函数 ---------- # 注释，表示以下是模型的训练函数定义。
def train_model(model, criterion, optimizer, num_epochs=10): # 定义一个名为 `train_model` 的函数，接受模型、损失函数、优化器和训练轮数作为参数。
    for epoch in range(num_epochs): # 开始训练循环，每个 `epoch` 遍历整个数据集一次。
        print(f"\nEpoch {epoch+1}/{num_epochs}") # 打印当前是第几个 epoch。
        print("-" * 30) # 打印一条分隔线，美化输出。

        for phase in ['train', 'val']: # 在每个 epoch 中，我们分别进行训练阶段和验证阶段。
            if phase == 'train': # 如果当前是训练阶段：
                model.train() # 将模型设置为训练模式。这会启用像 Dropout 和 BatchNorm 这样的层，它们在训练和评估时行为不同。
            else: # 如果当前是验证阶段：
                model.eval() # 将模型设置为评估模式。这会关闭 Dropout，并使用 BatchNorm 层的统计均值和方差，而不是批次统计。

            running_loss, running_corrects = 0.0, 0 # 初始化当前阶段的累计损失和正确预测数。

            for inputs, labels in dataloaders[phase]: # 遍历当前阶段的 DataLoader，逐批次获取输入图像和对应的真实标签。
                inputs, labels = inputs.to(device), labels.to(device) # 将输入图像和标签数据移动到指定的设备（GPU 或 CPU）上，以便进行计算。

                optimizer.zero_grad() # **重要步骤：** 在每次反向传播之前，清零所有参数的梯度。
                # PyTorch 默认会累积梯度，如果不清零，上一次的梯度会和当前的梯度叠加，导致错误的结果。

                with torch.set_grad_enabled(phase == 'train'): # 这是一个上下文管理器。
                    # 如果 `phase == 'train'` (训练阶段)，`torch.set_grad_enabled(True)` 会启用梯度计算（默认行为）。
                    # 如果 `phase == 'val'` (验证阶段)，`torch.set_grad_enabled(False)` 会禁用梯度计算。
                    # 在验证阶段禁用梯度计算可以节省内存并加速推理，因为此时我们不需要计算梯度来更新权重。
                    outputs = model(inputs) # **前向传播：** 将输入数据 `inputs` 传入模型，得到模型的输出 `outputs`（即 logit）。
                    _, preds = torch.max(outputs, 1) # 从模型的输出中获取预测结果。`torch.max(outputs, 1)` 返回每行（每个样本）的最大值和其对应的索引。`_` 表示我们不关心最大值本身，只关心索引（即预测的类别）。
                    loss = criterion(outputs, labels) # **计算损失：** 使用定义的损失函数 `criterion`，比较模型的 `outputs` 和真实的 `labels`，计算当前批次的损失。

                    if phase == 'train': # 如果当前是训练阶段：
                        loss.backward() # **反向传播：** 计算损失相对于模型所有可学习参数的梯度。
                        optimizer.step() # **更新权重：** 根据计算出的梯度和优化器规则，更新模型的权重。

                running_loss += loss.item() * inputs.size(0) # 累加当前批次的损失到 `running_loss`。`loss.item()` 获取单个损失值（Python 数值），`inputs.size(0)` 是当前批次的图像数量。
                running_corrects += (preds == labels).sum().item() # 累加当前批次中正确预测的数量。`(preds == labels)` 会生成一个布尔张量，`.sum()` 计算其中 True 的数量，`.item()` 转换为 Python 数值。

            epoch_loss = running_loss / dataset_sizes[phase] # 计算当前 epoch 的平均损失（总损失除以数据总数）。
            epoch_acc = running_corrects / dataset_sizes[phase] # 计算当前 epoch 的准确率（正确预测数除以数据总数）。
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}") # 打印当前阶段的损失和准确率，保留四位小数。

    return model # 训练完成后，返回训练好的模型。

# ---------- 启动训练 ---------- # 注释，表示以下是启动训练过程。
model = train_model(model, criterion, optimizer, num_epochs=10) # 调用 `train_model` 函数，开始模型的训练过程，这里设置为训练 10 个 epoch。训练好的模型会覆盖原来的 `model` 变量。

# ---------- 保存模型 ---------- # 注释，表示以下是保存模型的步骤。
model_path = os.path.join(model_dir, "resnet18_catdog.pth") # 构建模型保存的完整路径和文件名。
torch.save(model.state_dict(), model_path) # 保存模型的**状态字典**。`model.state_dict()` 包含了模型所有学习到的参数（权重和偏置）。
# 通常建议只保存状态字典，而不是整个模型对象，因为它更灵活，且更不容易受代码结构变化的影响。
print("Model saved to:", model_path) # 打印模型保存的路径。
