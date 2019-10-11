Segmentation Dataset
===

This repository contains a description of the DroneDeploy Segmentation Dataset and how to use it. It also contains example code to get a working segmentation model up and running quickly using a small sample dataset. See below for details of the full dataset and suggested improvement directions.

![Example](https://github.com/dronedeploy/dd-ml-segmentation-benchmark/raw/master/img/example.jpg)

### Quickstart

```
# clone this repository
git clone https://github.com/dronedeploy/dd-ml-segmentation-benchmark.git
cd dd-ml-segmentation-benchmark/
# install requirements
pip3 install -r requirements.txt
# optional: log in to W&B to see your training metrics,
# track your experiments, and submit your models to the benchmark
wandb login
# train Fastai model
python3 main.py
# train Keras model
python3 main_keras.py

```

### Training

To start training a model on a small sample dataset run the following, once working you should use the *full dataset*  by changing `main.py`

```
pip3 install -r requirements.txt
python3 main.py
```

This will download the sample dataset and begin training a model. You can monitor training performance on [Weights and Biases](https://www.wandb.com/). Once training is complete inference will be performed on all test scenes and a number of prediction images with names like `123123_ABCABC-prediction.png` will be created. After the the images are created they will be scored. Here's what the prediction looks like, not bad for 50 lines of code but there is a lot of room for improvement:  

![Example](https://github.com/dronedeploy/dd-ml-segmentation-benchmark/raw/master/img/out.gif)

### Dataset Details

The *full dataset* can be downloaded by changing a line in `main.py` this is the dataset that should be used for benchmarking. The dataset comprises a number of aerial scenes captured from drones. Each scene has a ground resolution of 10 cm per pixel. For each scene there is a corresponding "image", "elevation" and "label". The image is an RGB tif, the elevation is a single channel floating point .tif and the label is a PNG with 7 colors representing the 7 classes. Please see `index.csv` - inside the downloaded dataset - for a description of the quality of each labelled image and the distribution of the labels. To use the dataset you can split it into smaller chips (see `images2chips.py`). Here is an example of one of the labelled scenes:

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
- Data augmentation (`dataloader.py`)
- Hyper- parameters (`train.py`)
- Post-processing (`inference.py`)
- Chip size (`images2chips.py`)
- Model architecture (`train.py`)
- Elevation tiles are not currently used at all (`images2chips.py`)
