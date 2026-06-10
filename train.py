from model import *

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import os.path as osp


def train(model, device, train_loader, optimizer, epoch):
    # Switch to training mode: enable training-time behavior; mainly affects Dropout and BatchNorm, modify the `training` flag to `True` for current module and all sub-modules
    model.train()
    lossLayer = torch.nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(
        train_loader
    ):  # iter for all batches(train set)
        data, target = (
            data.to(device),
            target.to(device),
        )  # move data and target to the same device
        optimizer.zero_grad()  # clear grad, pytorch will accumulate grad from multiple epoch by default
        output = model(data)  # feed input data, get inference result
        loss = lossLayer(output, target)  # calc loss
        loss.backward()  # backpropogation
        optimizer.step()  # update parameters(weights) of model using SDG optimizer, momentum and calculated grad of last step

        if batch_idx % 50 == 0:
            print(
                "Train Epoch: {} [{}/{}]\tLoss: {:.6f}".format(
                    epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()
                )
            )


def test(model, device, test_loader):
    # Switch to evaluation(inference) mode: disable training-time behavior; mainly affects Dropout and BatchNorm, modify the `training` flag to `False` for current module and all sub-modules
    model.eval()
    test_loss = 0
    correct = 0
    lossLayer = torch.nn.CrossEntropyLoss(reduction="sum")
    for data, target in test_loader:  # iter for all batches(test set)
        data, target = data.to(device), target.to(device)
        output = model(data)  # inference
        test_loss += lossLayer(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        # calc the number of correct predictions
        # target.view_as(pred): adjust the shape of target same with pred
        # eq(): compare by element
        # get the equal quantity
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(
        test_loader.dataset
    )  # total_loss/amount of dataset to get avg loss

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {:.0f}%\n".format(
            test_loss, 100.0 * correct / len(test_loader.dataset)
        )
    )


if __name__ == "__main__":
    batch_size = 64  # for training, 64 pics for one batch
    test_batch_size = 64
    seed = 1
    epochs = 15
    lr = 0.01
    momentum = 0.5
    save_model = True
    using_bn = True

    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """
    MNIST: train sets: 60,000 pics
           test sets: 10,000 pics
    """
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=test_batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )

    if using_bn:
        model = NetBN().to(device)
    else:
        model = Net().to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    if save_model:
        if not osp.exists("ckpt"):
            os.makedirs("ckpt")
        if using_bn:
            torch.save(model.state_dict(), "ckpt/mnist_cnnbn.pt")
        else:
            torch.save(model.state_dict(), "ckpt/mnist_cnn.pt")
