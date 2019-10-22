Segmentation Dataset
===

This repository contains a description of the DroneDeploy Segmentation Dataset and how to use it. It also contains example code to get a working segmentation model up and running quickly using a small sample dataset. See below for details of the full dataset and suggested improvement directions.

![Example](https://github.com/dronedeploy/dd-ml-segmentation-benchmark/raw/master/img/example.jpg)

### Quickstart

Follow these steps to train a model and run inference end-to-end:

```
git clone https://github.com/dronedeploy/dd-ml-segmentation-benchmark.git
cd dd-ml-segmentation-benchmark
pip3 install -r requirements.txt

# optional: log in to W&B to track your experiements
wandb login

# train a Keras model
python3 main_keras.py

# train a Fastai model
python3 main_fastai.py
```

This will download the sample dataset and begin training a model. You can monitor training performance on [Weights & Biases](https://www.wandb.com/). Once training is complete, inference will be performed on all test scenes and a number of prediction images with names like `123123_ABCABC-prediction.png` will be created in the `wandb` directory. After the images are created they will be scored, and those scores stored in the `predictions` directory. Here's what a prediction looks like - not bad for 50 lines of code, but there is a lot of room for improvement:

![Example](https://github.com/dronedeploy/dd-ml-segmentation-benchmark/raw/master/img/out.gif)

### Dataset Details

The dataset comprises a number of aerial scenes captured from drones. Each scene has a ground resolution of 10 cm per pixel. For each scene there is a corresponding "image", "elevation" and "label". These are located in the `images`, `elevation` and `labels` directories.

The images are RGB TIFFs, the elevations are single channel floating point TIFFs (where each pixel value represents elevation in meters), and finally the labels are PNGs with 7 colors representing the 7 classes (documented below).

In addition please see `index.csv` - inside the downloaded dataset folder - for a description of the quality of each labelled image and the distribution of the labels.

To use a dataset for training, it must first be converted to chips (see `images2chips.py`). This will create two directories, `images-chips` and `label-chips`, which will contain a number of `300x300` (by default) RGB images. The `label-chips` are also RGB but will be very low pixel intensities `[0 .. 7]` so will appear black as first glance. You can use the `color2class` and `category2mask` function to switch between the two label representations.

Here is an example of one of the labelled scenes:

![Example](https://github.com/dronedeploy/dd-ml-segmentation-benchmark/raw/master/img/15efe45820_D95DF0B1F4INSPIRE-label.png)

Each color represents a different class.

Color (Blue, Green, Red) to Class Name:
---
```
(075, 025, 230) : BUILDING
(180, 030, 145) : CLUTTER
(075, 180, 060) : VEGETATION
(048, 130, 245) : WATER
(255, 255, 255) : GROUND
(200, 130, 000) : CAR
(255, 000, 255) : IGNORE
```

- IGNORE - These magenta pixels mask areas of missing labels or image boundaries. They can be ignored.

### Possible Improvements
----
The sample implementation is very basic and there is immediate opportunity to experiment with:
- Data augmentation (`datasets_keras.py`, `datasets_fastai.py`)
- Hyperparameters (`train_keras.py`, `train_fastai.py`)
- Post-processing (`inference_keras.py`, `inference_fastai.py`)
- Chip size (`images2chips.py`)
- Model architecture (`train_keras.py`, `train_fastai.py`)
- Elevation tiles are not currently used at all (`images2chips.py`)
