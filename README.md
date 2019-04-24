# Sketch-Artist

## Overview
A Tensorflow implementation of a Conditional GAN that generates human faces based on a description. This project was created for my CSC 340 (Artificial Intelligence) final project. Hi Kaur :)

## TO-DO
* Does not produce samples accurate to condition yet

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

## Label Vector
The label is a vector with 6 indices each corresponding to a different facial feature

The values for each feature can either be 1 or -1
* Black hair
* Blond hair
* Brown hair
* Glasses
* Male
* Beard

## Usage

```
python CGAN.py train
python CGAN.py test
```

## Results
### After 50 epochs
![epoch_50](https://user-images.githubusercontent.com/36581610/56628543-3969e400-6618-11e9-8438-12fa05be2e42.jpg)

![celebA](https://user-images.githubusercontent.com/36581610/56629037-150f0700-661a-11e9-89bd-ebfa1f8aa0a4.gif)
