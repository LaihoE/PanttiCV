import torchvision
import torch
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
def visualize(path,model_path):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_dataset_path=r'C:\Users\emill\PycharmProjects\canscanner/training'
    test_dataset_path=r'C:\Users\emill\PycharmProjects\canscanner/testing'
    custom_dataset_path=path

    train_transforms = transforms.Compose([transforms.Resize((224,224)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(20),
                                          transforms.ToTensor()])

    test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                          transforms.ToTensor()])

    train_dataset=torchvision.datasets.ImageFolder(root=train_dataset_path,transform=train_transforms)
    test_dataset=torchvision.datasets.ImageFolder(root=test_dataset_path,transform=test_transforms)
    custom_dataset=torchvision.datasets.ImageFolder(root=custom_dataset_path,transform=test_transforms)


    def show_transformed_images(dataset):
        loader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=True)
        batch = next(iter(loader))
        images, labels = batch

        grid = torchvision.utils.make_grid(images, nrow=3)
        plt.figure(figsize=(11, 11))
        plt.imshow(np.transpose(grid, (1, 2, 0)))
        print(labels)


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    custom_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=1, shuffle=False)



    model = torch.load(model_path)
    model.eval()


    def check_accuracy(loader, model):
        num_correct = 0
        num_samples = 0
        model.eval()

        with torch.no_grad():
            for x, y in loader:
                xd = x.to(device=device)
                yd = y.to(device=device)


                scores = model(xd)
                _, predictions = scores.max(1)

                num_correct += (predictions == yd).sum()
                num_samples += predictions.size(0)

                if predictions.item() == 1:
                    can="40 cent"
                else:
                    can="15 cent"

                if predictions != yd:
                    plt.imshow(x[0].permute(1, 2, 0))
                    plt.title([can, round(scores[0][0].item(), 2), round(scores[0][1].item(), 2)])
                    plt.show()


            print(
                f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
            )

        model.train()
    check_accuracy(custom_loader,model)


data_path=r'C:\Users\emill\PycharmProjects\canscanner/testing'
model_path="googlenetmodel.pt"
visualize(data_path,model_path)