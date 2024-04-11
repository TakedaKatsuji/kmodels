from VGG.model import vgg16
from VGG.dataset import VGGDataset
from VGG.utils import DataCsvParser
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

def train():
    pass

def validation():
    pass

def main():
    csv_path = "./datasets/COVID/covid_data.csv"

    model = vgg16(num_classes=2)


    # 損失関数とオプティマイザの定義
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    log_dir = "./logs/vgg_log"
    writer = SummaryWriter(log_dir)

    train_dataset = VGGDataset(csv_path=csv_path, mode="train")
    test_dataset = VGGDataset(csv_path=csv_path, mode="test")

    train_dataloader = DataLoader(dataset=train_dataset, num_workers=4, shuffle=True, batch_size=32)
    test_dataloader = DataLoader(dataset=test_dataset, num_workers=4, shuffle=True, batch_size=32)

    EPOCH = 128
    for epoch in range(EPOCH):
        pass


if __name__=="__main__()":
    main()
