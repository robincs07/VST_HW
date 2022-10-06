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
import os
from train import MySportVGGNet

TEST_PATH="Data/test"


class TestLoader(Dataset):
    def __init__(self, path):
        self.imgs=sorted(os.listdir(path))
    
    def __len__(self):
        return len(os.listdir(TEST_PATH))

    def __getitem__(self, index):
        self.img = cv2.imread(TEST_PATH+"/"+self.imgs[index])
        return torchvision.transforms.ToTensor()(self.img), self.imgs[index]


def main():
    f = open("test.csv", "w")
    f.write("names,label\n")
    # Device
    device =torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)
    # load model
    model = MySportVGGNet().to(device)
    model.load_state_dict(torch.load("sport_model.pth"))

    test_dataset = TestLoader(TEST_PATH)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.eval()
    for x, img_name in tqdm(test_loader):
        with torch.no_grad():
            x=x.to(device)
            pred = model(x)
            _, idx = torch.max(pred, 1)
        f.write("{}, {}\n".format("".join(img_name), idx.item()))

    f.close()



if __name__=="__main__":
    main()