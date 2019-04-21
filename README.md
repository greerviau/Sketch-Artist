# Sketch-Artist

## Overview
A Tensorflow implementation of a Conditional GAN that generates human faces based on a description. This project was created for CSC 340 (Artificial Intelligence) final project. 

## Installation
Git clone the repository and ```cd``` into the directory
```
git clone https://github.com/greerviau/Sketch-Artist.git && cd Sketch-Artist
```
Download the CelebA dataset [here](https://www.kaggle.com/jessicali9530/celeba-dataset) and extract

In CGAN.py add data directory to CelebA object
* Make sure that directory contains ```list_attr_celeba.csv``` and ```img_align_celeba```
```
celebA = CelebA(output_size, channel, sample_size, batch_size, crop, data_dir=<path-to-data>)
```

## Usage

```
python CGAN.py train
python CGAN.py test <model-version>
```
