import os
from pathlib import Path

import torchvision, torch
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import cv2

import pandas as pd
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

from cls import classes

#cuda gpu 사용
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#랜덤 시드 고정
torch.manual_seed(777)

# GPU 사용 가능일 경우 랜덤 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
    
# 파일 경로 지정
# https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign 데이터 사용
data_dir = Path("./archive/Meta")
train_path = Path("./archive/Train")
test_path = Path("./archive/Test")

# 이미지 전처리
img_height = 30
img_width = 30
channels = 3   

# 카테고리 수 확인
NUM_CATEGORIES = len(os.listdir(train_path))

# plt.figure(figsize=(14,14))

# for i in range(NUM_CATEGORIES):
#     plt.subplot(7, 7, i+1)
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#     sign = list(train_path.glob(f'{i}/*'))[0]
#     img = Image.open(sign)
#     plt.imshow(img)
# plt.show()

# # 어떤 표지판의 이미지가 많은지 시각화
folders = os.listdir('./archive/Train')

train_num = []
class_num = []

for folder in folders:
    train_files = os.listdir(str(train_path) + '/'+ folder) #리스트로 가져오면 에러떠서 str로 변환
    train_num.append(len(train_files))
    class_num.append(classes[int(folder)])

# # 각각의 클래스의 이미지의 수에 기초해 데이터셋 분류하기
# zipped_lists =  zip(train_num, class_num)
# sorted_pairs = sorted(zipped_lists)
# tuples =  zip(*sorted_pairs) # sorted(정렬할 데이터), 새로운 정렬된 리스트로 만들어서 반환
# train_num, class_num = [ list(tuple) for tuple in tuples]

# # 시각화
# plt.figure(figsize = (21, 10))
# plt.bar(class_num, train_num)
# plt.xticks(class_num, rotation='vertical')
# plt.show()

learning_rate = 0.001
batch_size = 100
training_epochs = 15

# dataset 정의
train_img = datasets.ImageFolder(root='./archive/Train',
                                     transform=transforms.Compose([
                                     transforms.ToTensor()
                                     ])
                                    )

test_img = datasets.ImageFolder(root='./archive/Test',
                                    transform=transforms.Compose([
                                    transforms.ToTensor()
                                    ])
                                    )

train_loader = torch.utils.data.DataLoader(train_img,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=8)

test_loader = torch.utils.data.DataLoader(test_img,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=8)

#cnn 모델 설계
class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN,self).__init__()
        self.keep_prob = 0.5

        #첫 번쨰 층
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout2d(0.25)    
            )
        
        #두 번째 층
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout2d(0.25)
            )
        
        #세 번째 층
        # self.layer3 = torch.nn.Sequential(
        #     torch.nn.Conv2d(64, 128, Kernel_size=3),
        #     torch.nn.ReLU()
        # )
        # TypeError: Conv2d.__init__() got an unexpected keyword argument 'Kernel_size' 오류..?

        self.fc1 = torch.nn.Linear(4 * 4 * 128, 625)
        self.fc2 = torch.nn.Linear(625, 10)
        
    #전결합층(Fully-Connteced Layer)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        #out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc2(out)
        return out

model = CNN().to(device)

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

total_batch = len(train_loader)
print('총 배치의 수 : {}'.format(total_batch))

def train():
    for epoch in range(training_epochs):
        avg_cost = 0

        for X, Y in train_loader:
            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            hypothesis = model(X)
            cost = criterion(hypothesis, Y)
            cost.backward()
            optimizer.step()

            avg_cost += cost / total_batch
        
        print('[Epoch :  {:>4}] cost = {:>.9}'.format(epoch+1, avg_cost))

if __name__ == '__main__':
    train()