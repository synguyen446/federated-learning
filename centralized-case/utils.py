from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def load_model():
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
)

    model.compile(optimizer='adam', loss= 'categorical_crossentropy',metrics=['accuracy'])

    return model

def visualize(model,X_test,y_test):
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
    
def evaluate(model, X_test, y_test):
    accuracy, loss = model.evaluate(X_test,y_test, batch_size = 32)
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Loss: {loss:3f}")