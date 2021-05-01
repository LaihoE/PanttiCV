import torch

model = torch.load("torchmodel.pt")
model.eval()

import torchvision
import torch
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

train_dataset_path=r'C:\Users\emill\PycharmProjects\canscanner/training'
test_dataset_path=r'C:\Users\emill\PycharmProjects\canscanner/testing'
custom_dataset_path=r'C:\Users\emill\PycharmProjects\canscanner/custom'


custom_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor()])

custom_dataset=torchvision.datasets.ImageFolder(root=custom_dataset_path,transform=custom_transforms)

custom_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=1, shuffle=False)



def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:

            xd = x.to(device=device)
            yd = y.to(device=device)

            scores = model(xd)
            scr, predictions = scores.max(1)
            print(scr)
            num_correct += (predictions == yd).sum()
            num_samples += predictions.size(0)

            plt.imshow(x[0].permute(1, 2, 0))
            plt.title([predictions,scr])
            plt.show()

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

check_accuracy(custom_loader,model)