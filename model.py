import os
from pathlib import Path

import pandas as pd
from PIL import Image
import numpy as np

import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

import multiprocessing

# CUDA GPU 사용
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 랜덤 시드 고정
torch.manual_seed(777)

# GPU 사용 가능일 경우 랜덤 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# 파일 경로 지정
train_path = Path("./archive/Train")
test_path = Path("./archive/Test")
csv_path = Path("./archive/Test.csv")

# 이미지 전처리
img_height = 30
img_width = 30

# 훈련 데이터셋 로드
train_transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = ImageFolder(root=train_path, transform=train_transform)

# DataLoader 생성
batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

# 커스텀 테스트 데이터셋 클래스 생성
class CustomDataset(Dataset):
    def __init__(self, csv_file, root, transform=None):
        self.data_info = pd.read_csv(csv_file)  # CSV 파일을 읽어서 데이터프레임으로 저장
        self.root = root  # 이미지 파일들이 위치한 디렉토리 경로
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        img_filename = self.data_info.iloc[index, 7]  # 이미지 파일명 가져오기
        img_path = self.root / img_filename  # 이미지 파일명을 경로에 맞게 가공

        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        label = int(self.data_info.iloc[index, 6])  # 해당 이미지의 라벨(ClassId) 가져오기

        # 이미지 파일과 라벨 확인
        print(f"Image path: {img_path}, Label: {label}")

        return image, label
        
# 테스트 데이터셋 로드
test_transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_dataset = CustomDataset(root=test_path, csv_file=csv_path, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

# CNN 모델 설계
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )

        self.fc1 = nn.Linear(6 * 6 * 64, 512)
        self.fc2 = nn.Linear(512, 43)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

def train(model, criterion, optimizer, train_loader):
    model.train()  # 모델을 학습 모드로 설정
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % 10 == 0:
            print(f'Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
            running_loss = 0.0

def test(model, test_loader):
    model.eval()  # 모델을 평가 모드로 설정
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total * 100
    print('Test Accuracy: {:.2f}%'.format(accuracy))

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    # CNN 모델 초기화
    model = CNN().to(device)

    # 손실 함수와 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 학습 설정
    num_epochs = 30

    multiprocessing.freeze_support()

    for epoch in range(num_epochs):
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        train(model, criterion, optimizer, train_loader)
        test(model, test_loader)

    print('학습 완료')
