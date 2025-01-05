import pandas as pd
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np


class Count:
    def __init__(self):
        try:
            with open("count.txt", "r") as file:
                self.index = int(file.readline())
        except:
            self.index = 0
            with open("count.txt", "w+") as file:
                file.write(str(self.index))

    def get_index(self):
        return self.index

    def next_index(self):
        self.index += 1
        if self.index < 4:
            with open("count.txt", "w+") as file:
                file.write(str(self.index))
        else:
            self.index = 0
            with open("count.txt", "w+") as file:
                file.write(str(self.index))


def load_data():
    count = Count()
    index = count.get_index()
    count.next_index()
    training_sample = pd.read_csv(f"data/sign-language/train/sample{index}.csv")
    testing_sample = pd.read_csv(r"C:\Users\syngu\Documents\GitHub\federated-learning\data\sign-language\test\test_sample.csv")
    X_train, y_train = training_sample.drop(columns="label"), training_sample["label"]
    X_test, y_test = testing_sample.drop(columns="label"), testing_sample['label']
    return prep_data(X_train,X_test,y_train,y_test)


def load_model():
    model = tf.keras.Sequential(
        [
            Conv2D(256, (3, 3), activation='leaky_relu',
                   input_shape=(28, 28, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.3),

            Conv2D(128, (3, 3), activation='leaky_relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.2),
            Flatten(),

            Dense(128, activation="leaky_relu"),

            Dense(25, activation='softmax')
        ]
    )
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy", metrics=['accuracy'])
    return model


def prep_data(X_train,X_test, y_train, y_test):
    scaler= StandardScaler()
    
    y_train = to_categorical(y_train, 25)
    y_test = to_categorical(y_test, 25)
    
    X_train =scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    

    imgs_train = []
    imgs_test =[]

    for i in range(len(X_train)):
        pixel_data = X_train[i]
        image = np.array(pixel_data, dtype=np.float32).reshape((28, 28))
        imgs_train.append(image)
        
    for i in range(len(X_test)):
        pixel_data = X_test[i]
        image = np.array(pixel_data, dtype=np.float32).reshape((28, 28))
        imgs_test.append(image)

    X_train = np.array(imgs_train).reshape(-1, 28, 28, 1)
    X_test = np.array(imgs_test).reshape(-1, 28, 28, 1)


    return X_train,X_test,y_train, y_test

def visualize(model, X_train):
    y_pred = model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=1)
    y_true = y_test.argmax(axis=1)
    
    
    cm = confusion_matrix(y_true,y_pred_classes)
    unique_labels = sorted(set(y_true) | set(y_pred_classes))
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels= unique_labels, yticklabels=unique_labels)
    plt.xlabel('Predicted value')
    plt.ylabel('Expected value')
    plt.title('Confusion Matrix')
    plt.show()
    
