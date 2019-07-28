# GAN with Gradient Reverasal Layer
Implemented with Chainer

Ganin, Yaroslav, et al. "Domain-adversarial training of neural networks." The Journal of Machine Learning Research 17.1 (2016): 2096-2030.

## Requirements
Chainer, OpenCV

```bash
$ pip install chainer opencv-python
```


## Training of GAN
```bash
$ python gan.py [options]
```

## Image generation from trained generator
```bash
$ python generate.py [options]
```

You can read help with `-h` option.

```bash
$ python gan.py -h
usage: gan.py [-h] [-b B] [-z Z] [-e E] [-r R] [--save_model]

Trains GAN

optional arguments:
  -h, --help    show this help message and exit
  -b B          batch size
  -z Z          dimension
  -e E          epoch
  -r R          result directory
  --save_model  save models
  
$ python generate.py -h
usage: generate.py [-h] -m M [-n N] [-z Z] [-r R]

Generates images randomly from trained generator model

optional arguments:
  -h, --help  show this help message and exit
  -m M        generator model file
  -n N        number of images to generate
  -z Z        dimension
  -r R        result directory
```
