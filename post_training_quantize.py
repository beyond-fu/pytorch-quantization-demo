from torch.serialization import load
from model import *

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import os.path as osp


# calc scale/zp for 500 batches, for freezing parameters later
def direct_quantize(model, test_loader):
    for i, (data, target) in enumerate(test_loader, 1):
        output = model.quantize_forward(data)  # calc min/max/scale/zp for a batch once
        if i % 500 == 0:  # iter for 500 batches
            break
    print("direct quantization finish")


def full_inference(model, test_loader):
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print(
        "\nTest set: Full Model Accuracy: {:.0f}%\n".format(
            100.0 * correct / len(test_loader.dataset)
        )
    )


def quantize_inference(model, test_loader):
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        output = model.quantize_inference(data)  # inference a batch once
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print(
        "\nTest set: Quant Model Accuracy: {:.0f}%\n".format(
            100.0 * correct / len(test_loader.dataset)
        )
    )


if __name__ == "__main__":
    batch_size = 64
    using_bn = True
    load_quant_model_file = None
    # load_model_file = None

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
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    if using_bn:
        model = NetBN()
        model.load_state_dict(torch.load("ckpt/mnist_cnnbn.pt", map_location="cpu"))
        save_file = "ckpt/mnist_cnnbn_ptq.pt"
    else:
        model = Net()
        model.load_state_dict(torch.load("ckpt/mnist_cnn.pt", map_location="cpu"))
        save_file = "ckpt/mnist_cnn_ptq.pt"

    model.eval()
    full_inference(model, test_loader)

    num_bits = 8
    model.quantize(num_bits=num_bits)  # initialize, prepare to quantize
    model.eval()
    print("Quantization bit: %d" % num_bits)

    if load_quant_model_file is not None:
        model.load_state_dict(torch.load(load_quant_model_file))
        print("Successfully load quantized model %s" % load_quant_model_file)

    direct_quantize(model, train_loader)  # exec quantize

    torch.save(model.state_dict(), save_file)
    model.freeze()

    # 测试是否设备转移是否正确
    # model.cuda()
    # print(model.qconv1.M.device)
    # model.cpu()
    # print(model.qconv1.M.device)

    quantize_inference(model, test_loader)
