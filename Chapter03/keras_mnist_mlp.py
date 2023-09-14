print("Classifying MNIST with a fully-connected Keras network with one hidden layer")

import tensorflow as tf

(X_train, Y_train), (X_validation, Y_validation) = \
    tf.keras.datasets.mnist.load_data()

X_train = X_train.reshape(60000, 784) / 255
X_validation = X_validation.reshape(10000, 784) / 255

classes = 10
Y_train = tf.keras.utils.to_categorical(Y_train, classes)
Y_validation = tf.keras.utils.to_categorical(Y_validation, classes)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation

input_size = 784
hidden_units = 100

model = Sequential([
    Dense(
        hidden_units, input_dim=input_size),
    BatchNormalization(),
    Activation('relu'),
    Dense(classes),
    Activation('softmax')
])

model.compile(
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    optimizer='adam')

model.fit(X_train, Y_train,
          batch_size=100, epochs=20,
          verbose=1)

score = model.evaluate(X_validation, Y_validation, verbose=1)
print('Validation accuracy:', score[1])

weights = model.layers[0].get_weights()

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy

fig = plt.figure()

w = weights[0].T
for unit in range(hidden_units):
    ax = fig.add_subplot(10, 10, unit + 1)
    ax.axis("off")
    ax.imshow(numpy.reshape(w[unit], (28, 28)), cmap=cm.Greys_r)

plt.savefig("unit_images.png", dpi=300)
plt.show()
