# Sign Language Translator  
This is a two part project. The first part will be developing the model (delivered); The second part will be developing a website so that user can use the model (not deliver).  
## Introduction
I developed a deep learning model with Python and Tensorflow to interpret sign language using Convolutional Neural Network - 2 convolutional layers, 1 regular layer, and 1 ouput layer. The model is trained on 8000 samples, and then validated with a different dataset, achieving more than 90% accuracy. 
## Installation
Make sure pip is installed using, for more information go [here](https://pip.pypa.io/en/stable/installation/)
Install packages.
```bash
pip install tensorflow 
```
Tensorflow is used for creating model and training model.
```bash
pip install scikit-learn
```
Scikit-learn is used For standardizing the data and generating confusion matrix.
```bash
pip install seaborn
pip install matplotlib
```
Seaborn and Mathplotlib is used For visualization and heatmap.
```bash
pip install pandas
pip install numpy
```
Pandas for reading csv and data processing.
Numpy for transforming csv data into a 28x28 images.
## Dataset
This project uses the [Sign Language Dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) from Kaggle, created by [tecperson](https://www.kaggle.com/datamunge).  
### License
This dataset is made available under the [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/) license. Please refer to the [original dataset page](https://www.kaggle.com/datasets/datamunge/sign-language-mnist).
## Cenralized Case
## Federated Learning Case 
