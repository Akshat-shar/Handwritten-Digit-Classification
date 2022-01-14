import tensorflow as tf
import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from tensorflow.python.keras.metrics import accuracy
mnist = tf.keras.datasets.mnist

# split the data in training set as tuple
(x_train, y_train), (x_test, y_test) = mnist.load_data()


#normalizing x_train and x_test
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


#layers
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# as the no of epochs increases, chance of overfitting of the model on training data also increases
model.fit(x_train, y_train, epochs=10)

#printing accuracy and loss
loss, accuracy = model.evaluate(x_test, y_test)
model.save('mnist.h5')
print(accuracy)
print(loss)


for x in range(1, 5):
    # now we are going to read images it with open cv
    img = cv.imread(f'{x}.png')[:, :, 0] 
    
    # invert black to white in images so that model won't get confues
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print("-------***------")
    print("The predicted value is : ", np.argmax(prediction))
    print("-------***------")
    # change the color in black and white format
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
