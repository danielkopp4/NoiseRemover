# Noise Remover

DNN Keras Noise remover.

## Examples

Dataset uses 3 kinds of noise: gausisan, salt-pepper, and spceckled noise.
Images are 256 x 256.
All of the below images were in the seperate testing dataset. 

![example_1 image](https://raw.githubusercontent.com/danielkopp4/NoiseRemover/main/examples/ex_1.png?token=AGFMV654PCCMNUPXAEE6MXTAHHPXG)

![example_2 image](https://raw.githubusercontent.com/danielkopp4/NoiseRemover/main/examples/ex_2.png?token=AGFMV6YAT2POBPZCOBVNBD3AHHQBY)

![example_3 image](https://raw.githubusercontent.com/danielkopp4/NoiseRemover/main/examples/ex_3.png?token=AGFMV63266MP42ZSU6DQEG3AHHQCM)

## Pretrained models

There are two pretrained models, an average noise (as seen in the examples) and a high noise reduction model.
Both are in the models directory and can be loaded by editing the `load_model` in the `config` class of `train.py`.
