from tensorflow import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import random
from keras.utils import plot_model




def build_model(width, height, depth, classes):
    model = Sequential()
    inputShape = (height, width, depth)
    model.add(Conv2D(32, (5, 5), padding="same",
        input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


    return model

def load_dataset(path, n):
    X_data = []
    y_data = []

    files_ = os.listdir(path)
    random.shuffle(files_)

    files = files_[:n]
    for file in files:
        image = cv2.imread(os.path.join(path, file))
        image = cv2.resize( image.astype('float'), ( 256, 256 ), interpolation = cv2.INTER_CUBIC )
        X_data.append (image)

        if file[0:3] == 'cat':
            y_data.append([1, 0])
        elif file[0:3] == 'dog':
            y_data.append([0, 1])

    X_data = np.array(X_data)
    X_data = (2 * X_data / 255) - 1

    y_data = np.array(y_data)

    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.20, random_state=42)

    return X_train, X_test, y_train, y_test

def main():
    n = 20
    X_train, X_test, y_train, y_test = load_dataset('train', 10)
    width = X_train.shape[1]
    height = X_train.shape[2]
    depth = X_train.shape[3]
    classes = 2
    model = build_model(width, height, depth, classes)
    training_results = model.fit(X_train, y_train, epochs=10, batch_size=8)
    print training_results.history
    classes = model.predict(X_test, batch_size=4)
    loss_and_metrics = model.evaluate(X_test, y_test, batch_size=4)
    print
    print "The results are ----\n"
    print "classes\n"
    print classes
    print "loss, accurcy\n"
    print loss_and_metrics


    X_test = (255/2) * (X_test + 1)
    X_train = (255 / 2) * (X_train + 1)

    for index, img in enumerate(X_test):
        name = './images/' + str(index) + 'test' + 'model2' + 'exp4' + '.jpg'
        cv2.imwrite(name, img)
        print index, y_test[index]

    for index, img in enumerate(X_train):
        name = './images/' + str(index) + 'train' + 'model2' + 'exp4' +'.jpg'
        cv2.imwrite(name, img)

    plot_model(model, to_file='model.png', show_shapes = True)


if __name__ == '__main__':
    main()
