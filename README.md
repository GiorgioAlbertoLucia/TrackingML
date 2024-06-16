# TrackingML

## Description
This project is a study of a Fully Connected Nueral Network approach to the tracking competition TrackML proposed on Kaggle. The data used for training and test is available to this [link]((https://www.kaggle.com/competitions/trackml-particle-identification)). 
The proposed approach starts from the second place solution. The code is optimised to work on a laptop thanks to a hybrid lazy-eager data handling. This code will rely on having enough disk space to store the processed datasets (the additional space required is as much as the one occupied by the downloaded datasets). In a full lazy approach the RAM required is minimal (although a hybrid approach is suggested for a reasonable processing time).

## Installation
To install the necessary dependencies, run:
```
pip install -r requirements.txt
```

## Usage 
The repository provides some core classes that can be used in any custom pipeline.

* BinaryHandler: class to prepare the dataset for the binary classification. The main features are also loaded in a smaller dataset to investigate the data structure
* FullyConnectedClassifier: Fully Connected Neural Network used for the classification task 
* ClassifierHandler: class to manage the training and testing of the classifier
* WeightedBCELoss: custom loss 
* Geometry classes: classes to study the detector and module geometry and reproduce it in 3D space. A class to create a voxel grid for a possible Convlutional Neural Network implementation is also available
* TrackReconstruction: class used to reconstruct tracks from the FCNN output and to score the algorithm

In the notebook folder, different notebooks for (pre- and postprocessed) data visualization are available. 
In the script folder a pipeline for the neural network is available. 