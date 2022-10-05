from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
import torch
import torch.optim
from tqdm import tqdm
import cv2
import torchvision
import numpy as np
import matplotlib.pyplot as plt

N_EPOTH=80
BATCH_SIZE=128
LEARNING_RATE=1e-5
MODEL_NAME = "sport_model.pth"

class SportLoader(Dataset):
    def __init__(self, mode, transform=None):
        self.mode=mode
        self.sport=pd.read_csv("Data/{}.csv".format(self.mode))
        self.img_name = self.sport["names"]
        self.label = self.sport["label"]
        self.transform = torchvision.transforms.ToTensor()
        
    
    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, index):
        img_dir = "Data/"+self.mode+"/"+self.img_name[index]
        self.img = cv2.imread(img_dir)
        self.target = self.label[index]

        if self.transform:
            self.img = self.transform(self.img)

        return self.img, self.target

class SportVGGNet(nn.Module):
    def __init__(self):
        super(SportVGGNet, self).__init__()

        self.conv_block_1=nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block_2=nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block_3=nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block_4=nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_5=nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc_block = nn.Sequential(
            nn.Linear(in_features = 7*7*512, out_features = 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(in_features = 4096, out_features = 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(in_features = 4096, out_features = 10)
        )

    def forward(self, x):
        output = self.conv_block_1(x)
        output = self.conv_block_2(output)
        output = self.conv_block_3(output)
        output = self.conv_block_4(output)
        output = self.conv_block_5(output)
        output = self.avgpool(output)
        # flatten
        output = output.reshape(output.shape[0], -1)
        output = self.fc_block(output)
        return output


def main():
    # Device
    device =torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)
    # dataset
    train_dataset = SportLoader("train")
    val_dataset = SportLoader("val")
    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    #loss and optimizer
    model = SportVGGNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loss = []
    train_acc = []
    val_loss=[]
    val_acc=[]
    for epoch in tqdm(range(N_EPOTH), desc="EPOCH"):
        # training
        model.train(True)
        total_loss=0
        correct=0
        for x, y in tqdm(train_loader, leave=False, desc="train"):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss+=loss.item()
            _, idx =torch.max(pred, 1)
            correct += float(torch.sum(idx == y))
            loss.backward()
            optimizer.step()
        avg_acc = correct/len(train_loader.dataset)
        avg_loss = total_loss/(len(train_loader.dataset)/BATCH_SIZE)
        train_loss.append(avg_loss)
        train_acc.append(avg_acc)

        # validation
        model.eval()
        total_loss=0
        correct=0
        for x, y in tqdm(val_loader, leave=False, desc="val"):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)
            total_loss+=loss.item()
            _, idx =torch.max(pred, 1)
            correct += float(torch.sum(idx == y))
        avg_acc = correct/len(val_loader.dataset)
        avg_loss = total_loss/(len(val_loader.dataset)/BATCH_SIZE)
        val_loss.append(avg_loss)
        val_acc.append(avg_acc)

    # loss array
    train_loss = np.asarray(train_loss)
    train_acc = np.asarray(train_acc)
    val_loss = np.asarray(val_loss)
    val_acc = np.asarray(val_acc)
    epoch_num = np.arange(N_EPOTH)

    # plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_num, train_loss,label="train" )
    plt.plot(epoch_num, val_loss, label="validation")
    plt.title("bs={}, lr={}, loss_curve".format(BATCH_SIZE, LEARNING_RATE))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc = "upper right")

    plt.subplot(1, 2, 2)
    plt.plot(epoch_num, train_acc,label="train" )
    plt.plot(epoch_num, val_acc, label="validation")
    plt.title("bs={}, lr={}, acc_curve".format(BATCH_SIZE, LEARNING_RATE))
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend(loc = "upper left")

    plt.savefig("loss_acc.png")

    torch.save(model.state_dict(), MODEL_NAME)



            
if __name__ == "__main__":
    main()