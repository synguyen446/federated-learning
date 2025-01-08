# Sign Language Translator  
This is a two part project. The first part will be developing the model (delivered); The second part will be developing a website so that user can use the model (not deliver).  
## Introduction
I developed a deep learning model with Python and Tensorflow to interpret sign language using Convolutional Neural Network - 2 convolutional layers, 1 regular layer, and 1 ouput layer. The model is trained on 8000 samples, and then validated with a different dataset, achieving more than 90% accuracy.  
Additionally, federated learning is also added to maintain data privacy by keeping the training data on edge devices (phone, computer, laptop); The trained parameters will be aggregated and sent back to the centralized model, ultimately, increasing the sample size.
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
## Models
Both case is test and trained on the same dataset, with different partition. Both case will also be using the same deep learning model - two convolutional neural layer, one regular layer, and one ouput layer. The softmax ouput is chosen to correctly predict 25 labels.
```python
model = tf.keras.Sequential(
    [
        Conv2D(256,(3,3),activation = 'leaky_relu', input_shape = (28,28,1)),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.3),

        Conv2D(128,(3,3),activation = 'leaky_relu'),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.2),
        Flatten(),

        Dense(128, activation= "leaky_relu"),
    
        Dense(25,activation='softmax')

    ]
```

### Centralized Case
This implementation is the traditional machine learning approach, where the model is trained on 8000 samples, and test on a different dataset.
#### Procedure
**1. Import necessary packages.**
```python
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import utils
```
**2. Read in the data.**
```python
df = pd.read_csv(r"") # Your training data directory
df_test = pd.read_csv(r"") # Your testing data directory
X_train = df.drop(columns="label") 
y_train = df['label']
X_test = df_test.drop(columns= "label")
y_test= df_test["label"]
```
**3. Standardizing the data.**
   + Create a scaler object
   + For training data, we want to fit (calculating parameters such as mean, standard deviation, etc) and then transform (applying the parameters to scale the training data)
   + For testing data, we want to keep the same parameters as the training data. Thus, we will only transform. 
```python
scaler = StandardScaler()                 
X_train = scaler.fit_transform(X_train)   
X_test = scaler.transform(X_test)
```
**4. Transforming the pixels into the image.**
   + We will loop through every training/testing sample and transform them into an array object using numpy array method (np.array)
   + We then reshape them into a 28x28 image using .reshape((28,28))
   + Lastly, we will need to reshape one more time; This time there are 4 parameters:  
         *-1      : automatically adjust to sample size;  
          (28,28): image size;  
          1      : one channel, the image is gray scale*
```python
for i in range(len(X_train)):
    pixel_data = X_train[i]
    image = np.array(pixel_data, dtype=np.float32).reshape((28, 28))
    imgs_train.append(image)

for i in range(len(X_test)):
    pixel_data = X_test[i]
    image = np.array(pixel_data, dtype=np.float32).reshape((28, 28))
    imgs_test.append(image)

X_train  = np.array(imgs_train).reshape(-1, 28, 28, 1)
X_test  = np.array(imgs_test).reshape(-1, 28, 28, 1)
```
**5. Loading model**
    + We will now load the model from [utils](https://github.com/synguyen446/sign-language-translator/blob/main/centralized-case/utils.py) module.
    + We will train the model on 10 epochs (10 iterations), and 128 batch size.
    + We will save the model for future testing.
```python
model = utils.load_model()
model.fit(X_train,  y_train, epochs= 10, batch_size=128)
model.save("centralized-model.keras")
```
**6 Visualizing the result**
```python
utils.visualize(model,X_train)
```
### Federated Learning Case 
