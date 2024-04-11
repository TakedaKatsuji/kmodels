import sys
sys.path.append("../")
from VGG.model import vgg16
from VGG.dataset import VGGDataset
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from ray import tune
import torch
from typing import Dict
from omegaconf import DictConfig
import hydra
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms

SEED = 42

def set_seed(seed:int)-> None:
    # Set random seeds for reproducibility
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(config):
    set_seed(seed=SEED)
    # load dataset
    # データ拡張のためのtransformを定義
    transform = A.Compose([
        A.Resize(224,224),  # 64x64にリサイズ
        A.Normalize(mean=(0.1307,), std=(0.3081,)),  # 平均と標準偏差で正規化
        ToTensorV2()  # テンソルに変換
    ])
    train_dataset = VGGDataset(csv_path=config.csv_path, mode="train", transform=transform)
    train_dataloader = DataLoader(dataset=train_dataset, num_workers=4, shuffle=True, batch_size=config.batch_size)

    model = vgg16(num_classes=2)
    model.to("cuda")
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config.epoch):
        model.train()
        for data, label in train_dataloader:
            optimizer.zero_grad()
            output = model(data.to("cuda"))
            loss = criterion(output.to("cpu"), label)
            loss.backward()
            optimizer.step()
        print(epoch)


@hydra.main(config_name="vgg_config.yaml", config_path="../configs", version_base=None)
def main(config:DictConfig):
    train(config)

if __name__=="__main__":
    main()
