import random
import time

import numpy as np
import torch
import torchvision
from torchvision import transforms

from model import Model

SAVE_MODEL_PATH = "checkpoint/best_accuracy.pth"


def train(opt):
    device = torch.device("cuda:0" if opt.use_gpu and torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = Model().to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # data preparation
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size)
    testset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size)

    # training epoch loop
    best_eval_acc = 0
    start = time.time()
    for ep in range(opt.num_epoch):
        avg_loss = 0
        model.train()
        print(f"{ep + 1}/{opt.num_epoch} epoch start")

        # training mini batch
        for i, (imgs, labels) in enumerate(trainloader):
            imgs, labels = imgs.to(device), labels.to(device)

            preds = model(imgs)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

            if i > 0 and i % 100 == 0:
                print(f"loss:{avg_loss / 100:.4f}")
                avg_loss = 0

        # validation
        if ep % opt.valid_interval == 0:
            tp, cnt = 0, 0
            model.eval()
            for i, (imgs, labels) in enumerate(testloader):
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.no_grad():
                    preds = model(imgs)
                preds = torch.argmax(preds, dim=1)
                tp += (preds == labels).sum().item()
                cnt += labels.shape[0]
            acc = tp / cnt
            print(f"eval acc:{acc:.4f}")

            # save best model
            if acc > best_eval_acc:
                best_eval_acc = acc
                torch.save(model.state_dict(), SAVE_MODEL_PATH)

        print(f"{ep + 1}/{opt.num_epoch} epoch finished. elapsed time:{time.time() - start:.1f} sec")

    print(f"training finished. best eval acc:{best_eval_acc:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual_seed", type=int, default=1111, help="random seed setting")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--num_epoch", type=int, default=30, help="number of epochs to train")
    parser.add_argument("--valid_interval", type=int, default=1, help="validation interval")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--use_gpu", action="store_true", help="use gpu if available")
    opt = parser.parse_args()
    print("args", opt)

    # set seed
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)

    # training
    train(opt)
