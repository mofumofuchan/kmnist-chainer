# KMNIST-chainer

## requirements
- numpy
- chainer >= 6.0.0b1 (for chainerx)

## quick start
Download [KMNIST datasets](https://github.com/rois-codh/kmnist).
```
$ source download_kmnist.sh
```

Then train multi layer perceptron with KMNIST.

```
$ python train_kmnist.py
```

## reference
[1] Clanuwat et al, "Deep Learning for Classical Japanese Literature", NeurIPS 2018 Workshop on Machine Learning for Creativity and Design, 2018.

