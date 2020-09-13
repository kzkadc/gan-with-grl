# coding: utf-8

from train_gan import get_generator
import cv2
from pathlib import Path
import pprint
import numpy as np
import chainer
from chainer import serializers
chainer.config.use_ideep = "auto"
chainer.config.train = False


def parse_args():
    desc = """
    Generates images randomly from trained generator model
    """

    import argparse
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-m", required=True, help="generator model file")
    parser.add_argument("-n", type=int, default=10, help="number of images to generate")
    parser.add_argument("-z", type=int, default=64, help="dimension")
    parser.add_argument("-r", default="result", help="result directory")

    args = parser.parse_args()
    pprint.pprint(vars(args))
    main(args)


def main(args):
    gen = get_generator(args.z)
    serializers.load_npz(args.m, gen)
    gen.to_intel64()

    z = np.random.uniform(size=(args.n, args.z)).astype(np.float32)
    images = (gen(z).array.squeeze() * 255).astype(np.uint8)

    p = Path(args.r)
    try:
        p.mkdir(parents=True)
    except FileExistsError:
        pass

    for i, img in enumerate(images):
        cv2.imwrite(str(p / "img_{:03d}.png".format(i)), img)


if __name__ == "__main__":
    parse_args()
