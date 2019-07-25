# coding: utf-8

import argparse
import cv2
from pathlib import Path
import pprint
from functools import partial
import numpy as np
import chainer
from chainer import Variable, optimizers, Chain, iterators, optimizers, training
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L
chainer.config.use_ideep = "auto"


def parse_args():
    desc = """
    Trains GAN
    """

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-b", type=int, default=64, help="batch size")
    parser.add_argument("-z", type=int, default=64, help="dimension")
    parser.add_argument("-e", type=int, default=10, help="epoch")
    parser.add_argument("-r", default="result", help="result directory")
    parser.add_argument("--save_model", action="store_true", help="save models")

    args = parser.parse_args()
    pprint.pprint(vars(args))
    main(args)


def main(args):
    mnist_train = chainer.datasets.get_mnist(ndim=3, withlabel=False)[0]
    itr = iterators.SerialIterator(mnist_train, args.b, shuffle=True, repeat=True)

    model = GAN(args.z)
    if chainer.config.use_ideep != "never":
        model.to_intel64()
    opt = optimizers.Adam(alpha=0.0002, beta1=0.5, beta2=0.9)
    opt.setup(model)

    updater = training.StandardUpdater(itr, opt)
    trainer = training.Trainer(updater, stop_trigger=(args.e, "epoch"), out=args.r)

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(["main/loss_real", "main/loss_fake", "epoch"]))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.PlotReport(["main/loss_real", "main/loss_fake"], filename="loss.pdf"))
    trainer.extend(ext_save_img(model.gen, args.r, args.z))
    trainer.extend(extensions.DumpGraph("main/loss_fake"))
    if args.save_model:
        trainer.extend(extensions.snapshot_object(model.gen, "gen_epoch_{.updater.epoch:04d}.npz"), trigger=(10, "epoch"))
        trainer.extend(extensions.snapshot_object(model.dis, "dis_epoch_{.updater.epoch:04d}.npz"), trigger=(10, "epoch"))

    trainer.run()


def ext_save_img(generator, out, zdim):
    # extension to save generated images

    out_path = Path(out, "gen_img")
    try:
        out_path.mkdir(parents=True)
    except FileExistsError:
        pass

    @chainer.training.make_extension(trigger=(1, "epoch"))
    def _ext_save_img(trainer):
        z = np.random.uniform(size=(1, zdim)).astype(np.float32)
        with chainer.using_config("train", False):
            img = generator(Variable(z)).array * 255
        img = img.squeeze().astype(np.uint8)

        p = out_path / "out_epoch_{:03d}.png".format(trainer.updater.epoch)
        cv2.imwrite(str(p), img)

    return _ext_save_img


class GAN(Chain):
    def __init__(self, zdim):
        super().__init__()
        self.zdim = zdim
        with self.init_scope():
            self.gen = get_generator(zdim)
            self.dis = get_discriminator()

    def __call__(self, x_real):
        z = np.random.uniform(size=(len(x_real), self.zdim)).astype(np.float32)
        z = Variable(z)

        x_fake = self.gen(z)
        y_fake = self.dis(gradient_reversal_layer(x_fake))
        y_real = self.dis(x_real)

        loss_fake = F.mean(F.softplus(y_fake))
        loss_real = F.mean(F.softplus(-y_real))

        chainer.report({
            "loss_real": loss_real,
            "loss_fake": loss_fake
        }, self)

        return loss_fake + loss_real


N = 32


def get_discriminator():
    kwds = {
        "ksize": 4,
        "stride": 2,
        "pad": 1,
        "nobias": True
    }
    model = chainer.Sequential(
        L.Convolution2D(1, N, **kwds),
        L.BatchNormalization(N),
        F.relu,
        L.Convolution2D(N, N * 2, **kwds),
        L.BatchNormalization(N * 2),
        F.relu,
        L.Convolution2D(N * 2, N * 4, ksize=2, stride=1, pad=0, nobias=True),
        L.BatchNormalization(N * 4),
        F.relu,
        L.Convolution2D(N * 4, N * 8, **kwds),
        L.BatchNormalization(N * 8),
        F.relu,
        L.Convolution2D(N * 8, 1, ksize=1, stride=1, pad=0),
        partial(F.mean, axis=(1, 2, 3))
    )

    return model


def get_generator(zdim):
    kwds = {
        "ksize": 4,
        "stride": 2,
        "pad": 1,
        "nobias": True
    }
    model = chainer.Sequential(
        L.Linear(zdim, 3 * 3 * N * 8, nobias=True),
        partial(F.reshape, shape=(-1, N * 8, 3, 3)),
        L.BatchNormalization(N * 8),
        F.relu,
        L.Deconvolution2D(N * 8, N * 4, **kwds),
        L.BatchNormalization(N * 4),
        F.relu,
        L.Deconvolution2D(N * 4, N * 2, ksize=2, stride=1, pad=0, nobias=True),
        L.BatchNormalization(N * 2),
        F.relu,
        L.Deconvolution2D(N * 2, N, **kwds),
        L.BatchNormalization(N),
        F.relu,
        L.Deconvolution2D(N, 1, ksize=4, stride=2, pad=1),
        F.sigmoid
    )

    return model


class GRL(chainer.function_node.FunctionNode):
    def forward_cpu(self, inputs):
        return inputs

    def backward(self, inputs, grad_outputs):
        gy, = grad_outputs
        return - gy,


def gradient_reversal_layer(x):
    y, = GRL().apply((x,))
    return y


if __name__ == "__main__":
    parse_args()
