import sys

sys.path.append("../")
from VGG.dataset import VGGDataset
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from ray import tune
import torch
from typing import Dict
import hydra
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import models
import mlflow
from tqdm import tqdm

SEED = 42


def set_seed(seed: int) -> None:
    # Set random seeds for reproducibility
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@hydra.main(config_name="vgg_config.yaml", config_path="../configs", version_base=None)
def train(config):
    mlflow.start_run(run_name="VGG")
    mlflow.log_params(config)
    set_seed(seed=SEED)
    # load dataset
    # データ拡張のためのtransformを定義
    train_transform = A.Compose(
        [
            A.Resize(256, 256),  # 画像サイズをリサイズ
            A.RandomCrop(224, 224),  # ランダムクロップ
            A.HorizontalFlip(p=0.5),  # 水平方向にランダムに反転
            A.RandomBrightnessContrast(p=0.2),  # ランダムな明るさとコントラストの変化
            A.Rotate(limit=(-30, 30), p=0.5),  # ランダムな角度で回転
            A.Normalize(mean=(0.1307,), std=(0.3081,)),  # 平均と標準偏差で正規化
            ToTensorV2(),  # テンソルに変換
        ]
    )
    test_transform = A.Compose(
        [
            A.Resize(224, 224),  # 64x64にリサイズ
            A.Normalize(mean=(0.1307,), std=(0.3081,)),  # 平均と標準偏差で正規化
            ToTensorV2(),  # テンソルに変換
        ]
    )
    train_dataset = VGGDataset(csv_path=config.csv_path, mode="train", transform=train_transform)
    train_dataloader = DataLoader(
        dataset=train_dataset, num_workers=4, shuffle=True, batch_size=config.batch_size, drop_last=True
    )

    test_dataset = VGGDataset(csv_path=config.csv_path, mode="test", transform=test_transform)
    test_dataloader = DataLoader(dataset=test_dataset, num_workers=4, shuffle=False, batch_size=config.batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vgg_model = models.vgg16(pretrained=True)
    for param in vgg_model.parameters():
        param.requires_grad = False

    # 新しく追加した最後の層のパラメータのみを学習可能にします
    for param in vgg_model.classifier.parameters():
        param.requires_grad = True

    num_features = vgg_model.classifier[6].in_features
    features = list(vgg_model.classifier.children())[:-1]  # 最後の層を削除
    features.extend([nn.Linear(num_features, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1)])  # 新しい層を追加
    vgg_model.classifier = nn.Sequential(*features)

    vgg_model.to(device)
    optimizer = optim.Adam(vgg_model.parameters(), lr=config.lr)

    criterion = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(config.epoch)):
        vgg_model.train()
        running_loss = 0.0
        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.float().to(device)
            optimizer.zero_grad()
            outputs = vgg_model(inputs)
            loss = criterion(outputs.view(-1), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        # バッチごとの平均損失を計算
        epoch_loss = running_loss / len(train_dataset)

        # 検証データでの性能を評価
        vgg_model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs = inputs.to(device)
                labels = labels.float().to(device)
                outputs = vgg_model(inputs)
                loss = criterion(outputs.view(-1), labels)
                test_loss += loss.item() * inputs.size(0)
                predicted = torch.round(torch.sigmoid(outputs)).view(-1)
                correct += (predicted == labels).sum().item()
                total += len(labels)
        # 検証データでの正解率を計算
        test_acc = correct / total

        mlflow.log_metrics(
            {"train_loss": round(epoch_loss, 3), "test_loss": round(test_loss, 3), "test_acc": round(test_acc, 3)},
            step=epoch,
        )
        mlflow.pytorch.log_model(vgg_model, "model")

    mlflow.end_run()


if __name__ == "__main__":
    train()
