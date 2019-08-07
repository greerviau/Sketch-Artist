# Sketch-Artist

## Overview
A Tensorflow implementation of a Conditional GAN that generates human faces based on a description. This project was created for my CSC 340 (Artificial Intelligence) final project. Hi Kaur :)

### [YouTube Video](https://www.youtube.com/watch?v=KCtoZrOBZ7g&t)

## TO-DO
* Does not produce samples accurate to condition yet

## Label Vector
The label is a vector with 5 indices each corresponding to a different facial feature

The values for each feature can either be 1 or -1
* Black hair
* Blonde hair
* Brown hair
* Male
* Beard

## Generator

![Generator](https://user-images.githubusercontent.com/36581610/56740551-27cf2c00-673f-11e9-9459-ac9cfde16da1.png)

## Discriminator

![Discriminator](https://user-images.githubusercontent.com/36581610/56740614-48978180-673f-11e9-8e22-16d22ff39411.png)

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
python CGAN.py test
```

## Results
### After 100 epochs
![epoch_100_batch_310](https://user-images.githubusercontent.com/36581610/61691196-014a6e00-acf9-11e9-88e6-9bd8e48db169.jpg)
