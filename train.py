import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# 1. GPU 장치 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 하이퍼파라미터
BATCH_SIZE = 32
EPOCHS = 30
DATA_DIR = './PokemonData'

# 2. 데이터 전처리 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. 데이터셋 로드 및 자동 분할
if not os.path.exists(DATA_DIR):
    print(f"Error: '{DATA_DIR}' 폴더를 찾을 수 없습니다. 폴더명을 확인하세요.")
else:
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    
    # 80% 학습, 20% 검증 분할
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    NUM_CLASSES = len(full_dataset.classes)
    print(f"발견된 포켓몬 종류: {NUM_CLASSES}개")

# 4. 실험 설정 리스트
configs = [
    {"name": "Exp1_ResNet18_FineTune", "model": "resnet18", "pretrained": True, "freeze": False},
    {"name": "Exp2_ResNet18_Freeze", "model": "resnet18", "pretrained": True, "freeze": True},
    {"name": "Exp3_MobileNet_FineTune", "model": "mobilenet", "pretrained": True, "freeze": False},
    {"name": "Exp4_ResNet18_Scratch", "model": "resnet18", "pretrained": False, "freeze": False},
]

def train_model(config, train_loader, val_loader):
    print(f"\n실험 시작: {config['name']}")
    os.makedirs(config['name'], exist_ok=True)
    
    # 모델 설정
    if config['model'] == "resnet18":
        model = models.resnet18(pretrained=config['pretrained'])
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    else:
        model = models.mobilenet_v2(pretrained=config['pretrained'])
        model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
    
    model = model.to(device)

    # 가중치 고정(Freeze) 로직
    if config['freeze']:
        for param in model.parameters():
            param.requires_grad = False
        target_layer = model.fc if config['model'] == "resnet18" else model.classifier[1]
        for param in target_layer.parameters():
            param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

    history = {'train_loss': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            pbar.set_postfix(loss=loss.item())
            
        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        history['train_loss'].append(epoch_loss)
        history['val_acc'].append(val_acc)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # 결과 저장
    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.legend()
    plt.title(f"Learning Curve - {config['name']}")
    plt.savefig(f"{config['name']}/learning_curve.png")
    plt.close()
    
    torch.save(model.state_dict(), f"{config['name']}/model.pth")

# 실행 루프
if __name__ == "__main__":
    for config in configs:
        train_model(config, train_loader, val_loader)
    print("\n모든 실험이 완료되었습니다!")