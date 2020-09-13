from train_gan import get_generator

import torch

import cv2
import numpy as np

from pathlib import Path
import pprint


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
    parser.add_argument("-g", type=int, default=-1, help="GPU id (negative value indicates CPU)")

    args = parser.parse_args()
    pprint.pprint(vars(args))
    main(args)


def main(args):
    if args.g >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.g:d}")
        print(f"GPU mode: {args.g:d}")
    else:
        device = torch.device("cpu")
        print("CPU mode")

    gen = get_generator(args.z)
    gen.load_state_dict(torch.load(args.m))
    gen.to(device)

    z = np.random.uniform(size=(args.n, args.z)).astype(np.float32)
    z = torch.from_numpy(z).to(device)
    images = (gen(z).detach().cpu().numpy().squeeze() * 255).astype(np.uint8)

    p = Path(args.r)
    try:
        p.mkdir(parents=True)
    except FileExistsError:
        pass

    for i, img in enumerate(images):
        cv2.imwrite(str(p / "img_{:03d}.png".format(i)), img)


if __name__ == "__main__":
    parse_args()
