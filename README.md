# Sketch-Artist

## Overview
A Conditional GAN that generates faces based on a description

## TO-DO
* Increase sample size (currently 1000)
* Improve model architecture for generator and discriminator

## Instalation
Git clone the repository and ```cd``` into the directory
```
git clone https://github.com/greerviau/Sketch-Artist.git && cd Sketch-Artist
```
Download the CelebA dataset [here](https://www.kaggle.com/jessicali9530/celeba-dataset) and extract

In CGAN.py add data directory to CelebA object
```
celebA = CelebA(output_size, channel, sample_size, batch_size, <path-to-data>)
```

## Usage
```
python CGAN.py train
python CGAN.py test <model-version>
```
