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

plt.figure(figsize=(14,14))
idx = 0

for i in range(NUM_CATEGORIES):
    plt.subplot(7, 7, i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    sign = list(train_path.glob(f'{i}/*'))[0]
    img = Image.open(sign)
    plt.imshow(img)
plt.show()

# # 어떤 표지판의 이미지가 많은지 시각화
folders = os.listdir('./archive/Train')

train_num = []
class_num = []

for folder in folders:
  train_files = os.listdir(str(train_path) + '/'+ folder) #리스트로 가져오면 에러떠서 str로 변환
  train_num.append(len(train_files))
  class_num.append(classes[int(folder)])

# 각각의 클래스의 이미지의 수에 기초해 데이터셋 분류하기
zipped_lists =  zip(train_num, class_num)
sorted_pairs = sorted(zipped_lists)
tuples =  zip(*sorted_pairs) # sorted(정렬할 데이터), 새로운 정렬된 리스트로 만들어서 반환
train_num, class_num = [ list(tuple) for tuple in tuples]

# # 시각화
plt.figure(figsize = (21, 10))
plt.bar(class_num, train_num)
plt.xticks(class_num, rotation='vertical')
plt.show()

# dataset 정의
# for i in range(NUM_CATEGORIES):
#   sign = list(train_path.glob(f'{i}/*'))[0]

# image_forlder = datasets.ImageFolder(root=sign,
#                                      transform=transforms.Compose([
#                                      transforms.ToTensor()
#                                      ])
#                                     )

# train_loader = torch.utils.data.DataLoader(image_forlder,
#                                            batch_size=50,
#                                            shuffle=True,
#                                            num_workers=8)

# test_loader = torch.utils.data.DataLoader(test_path,
#                                           batch_size=50,
#                                           shuffle=False,
#                                           num_workers=8)

# images, labels = next(iter(train_loader))
