from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
import tensorflow as tf

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
