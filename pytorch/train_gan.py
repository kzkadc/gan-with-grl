
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data
from torchvision.datasets import MNIST
from ignite.engine.engine import Engine
from ignite.engine import Events

import numpy as np
import matplotlib.pyplot as plt
import cv2

import pprint
from pathlib import Path


def parse_args():
    import argparse

    desc = """
    Trains GAN
    """

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-b", type=int, default=64, help="batch size")
    parser.add_argument("-z", type=int, default=64, help="dimension")
    parser.add_argument("-e", type=int, default=100, help="epoch")
    parser.add_argument("-r", default="result", help="result directory")
    parser.add_argument("--save_model", action="store_true", help="save models")
    parser.add_argument("-g", type=int, default=-1, help="GPU id (negative value indicates CPU)")

    args = parser.parse_args()
    pprint.pprint(vars(args))
    main(args)


def main(args):
    if args.g >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.g:d}")
        print(f"GPU mode: {args.g}")
    else:
        device = torch.device("cpu")
        print("CPU mode")

    result_dir = Path(args.r)
    try:
        result_dir.mkdir(parents=True)
    except FileExistsError:
        pass

    mnist_train = MNIST(root=".", train=True, download=True,
                        transform=lambda x: np.expand_dims(np.asarray(x, dtype=np.float32), 0) / 255)
    train_loader = data.DataLoader(mnist_train, batch_size=args.b, shuffle=True)

    model = GAN(args.z).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.9))

    trainer = Engine(GANTrainer(model, opt, device))
    logger = GANLogger(model, train_loader, device)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, logger)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, print_loss(logger))
    trainer.add_event_handler(Events.EPOCH_COMPLETED, plot_loss(logger, result_dir / "loss.pdf"))
    trainer.add_event_handler(Events.EPOCH_COMPLETED, save_img(model.gen, result_dir / "output_images", args.z, device))
    if args.save_model:
        trainer.add_event_handler(Events.EPOCH_COMPLETED, save_model(model, result_dir / "models"))

    trainer.run(train_loader, max_epochs=args.e)


class GANTrainer:
    def __init__(self, gan, opt, device):
        self.gan = gan
        self.opt = opt
        self.device = device

    def __call__(self, engine, batch):
        self.gan.train()
        self.opt.zero_grad()

        x_real, _ = batch
        x_real = x_real.to(self.device)

        loss_fake, loss_real = self.gan(x_real)
        loss = loss_fake + loss_real
        loss.backward()
        self.opt.step()

        return {
            "loss_fake": loss_fake.item(),
            "loss_real": loss_real.item()
        }


class GANLogger:
    def __init__(self, gan, train_loader, device):
        self.gan = gan
        self.train_loader = train_loader
        self.device = device

        self.epochs = []
        self.loss_real = []
        self.loss_fake = []

    def __call__(self, engine):
        self.gan.eval()
        n = 0
        loss_fake_mean, loss_real_mean = 0, 0
        for batch in self.train_loader:
            x, _ = batch
            x = x.to(self.device)
            loss_fake, loss_real = self.gan(x)
            loss_fake_mean += loss_fake.item() * len(batch)
            loss_real_mean += loss_real.item() * len(batch)
            n += len(batch)

        self.epochs.append(engine.state.epoch)
        self.loss_fake.append(loss_fake_mean / n)
        self.loss_real.append(loss_real_mean / n)


def print_loss(logger):
    def _report(engine):
        print(f"epoch {logger.epochs[-1]:d}, loss_fake: {logger.loss_fake[-1]:f}, loss_real: {logger.loss_real[-1]:f}")

    return _report


def plot_loss(logger, file_path):
    def _plot(engine):
        plt.figure()
        plt.plot(logger.epochs, logger.loss_fake, label="loss_fake")
        plt.plot(logger.epochs, logger.loss_real, label="loss_real")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(str(file_path))
        plt.close()

    return _plot


def save_img(generator: nn.Module, out_dir_path: Path, zdim: int, device):
    if isinstance(out_dir_path, str):
        out_dir_path = Path(out_dir_path)
    try:
        out_dir_path.mkdir(parents=True)
    except FileExistsError:
        pass

    def _save(engine):
        generator.eval()

        z = np.random.uniform(size=(1, zdim)).astype(np.float32)
        z = torch.from_numpy(z).to(device)
        with torch.no_grad():
            img = generator(z)
        img = img.cpu().detach().numpy().squeeze() * 255
        img = img.astype(np.uint8)

        p = out_dir_path / f"out_epoch_{engine.state.epoch:03d}.png"
        cv2.imwrite(str(p), img)

    return _save


def save_model(gan, out_dir_path):
    if isinstance(out_dir_path, str):
        out_dir_path = Path(out_dir_path)
    try:
        out_dir_path.mkdir(parents=True)
    except FileExistsError:
        pass

    def _save(engine):
        p = out_dir_path / f"gen_epoch-{engine.state.epoch:03d}.pt"
        torch.save(gan.gen.state_dict(), str(p))
        p = out_dir_path / f"dis_epoch-{engine.state.epoch:03d}.pt"
        torch.save(gan.dis.state_dict(), str(p))

    return _save


class GAN(nn.Module):
    def __init__(self, zdim, device="cpu"):
        super().__init__()
        self.zdim = zdim
        self.device = device
        self.gen = get_generator(zdim)
        self.dis = get_discriminator()

    def forward(self, x_real):
        z = np.random.uniform(size=(len(x_real), self.zdim)).astype(np.float32)
        z = torch.from_numpy(z).to(self.device)

        x_fake = self.gen(z)
        y_fake = self.dis(gradient_reversal_layer(x_fake))
        y_real = self.dis(x_real)

        loss_fake = F.softplus(y_fake).mean()
        loss_real = F.softplus(-y_real).mean()

        return loss_fake, loss_real


N = 32


def get_discriminator():
    kwds = {
        "kernel_size": 4,
        "stride": 2,
        "padding": 1,
        "bias": False
    }
    return nn.Sequential(
        nn.Conv2d(1, N, **kwds),
        nn.BatchNorm2d(N),
        nn.ReLU(inplace=True),
        nn.Conv2d(N, N * 2, **kwds),
        nn.BatchNorm2d(N * 2),
        nn.ReLU(inplace=True),
        nn.Conv2d(N * 2, N * 4, kernel_size=2, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(N * 4),
        nn.ReLU(inplace=True),
        nn.Conv2d(N * 4, N * 8, **kwds),
        nn.BatchNorm2d(N * 8),
        nn.ReLU(inplace=True),
        nn.Conv2d(N * 8, 1, kernel_size=1, stride=1, padding=0),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten()
    )


def get_generator(zdim):
    kwds = {
        "kernel_size": 4,
        "stride": 2,
        "padding": 1,
        "bias": False
    }
    return nn.Sequential(
        nn.Linear(zdim, 3 * 3 * N * 8, bias=False),
        Lambda(lambda x: x.reshape(-1, N * 8, 3, 3)),
        nn.BatchNorm2d(N * 8),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(N * 8, N * 4, **kwds),
        nn.BatchNorm2d(N * 4),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(N * 4, N * 2, kernel_size=2, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(N * 2),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(N * 2, N, **kwds),
        nn.BatchNorm2d(N),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(N, 1, kernel_size=4, stride=2, padding=1),
        nn.Sigmoid()
    )


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return - grad_output


def gradient_reversal_layer(x):
    return GradientReversalLayer.apply(x)


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


if __name__ == "__main__":
    parse_args()
