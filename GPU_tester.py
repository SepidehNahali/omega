
from keras.models import Sequential
from keras.layers import *
from keras.datasets import *
from keras.losses import *
from keras.optimizers import *
import keras
from keras import backend as K
import numpy as np
from keras import utils as np_utils

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train)  # 60000*28*28
# print(y_train)  # 60000
# print(x_test)  # 10000*28*28
# print(y_test)  # 10000
# print(K.image_data_format())
# exit(1)
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')
else:  # tf是channels_last
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

x_train /= 255
x_test /=255

y_train = keras.utils.np_utils.to_categorical(y_train, 10)
y_test = keras.utils.np_utils.to_categorical(y_test, 10)
# print(y_train.shape)
# print(y_test.shape)
# exit(1)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten()) 
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation=keras.activations.softmax))

# model.summary()
# exit(1)

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=128, epochs=100, verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=1)
print(score)

