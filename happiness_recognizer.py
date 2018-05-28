
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model, Sequential
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input
from keras import optimizers
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import keras.backend as K

# K.set_image_data_format('channels_last')

import h5py
import os

class HappinessRecognizer:
  def __init__(self, img_path):
    self.dir_path = os.path.dirname(os.path.realpath(__file__))
    self.img_path = os.path.join(self.dir_path, img_path)

    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, _ = self.load_dataset()
    # Normalize image vectors
    X_train = X_train_orig/255.
    X_test = X_test_orig/255.

    # Reshape
    Y_train = Y_train_orig.T
    Y_test = Y_test_orig.T

    saved_model_path = os.path.join(self.dir_path, 'model.h5')
    if os.path.exists(saved_model_path):
      model = load_model(saved_model_path)
      model.summary()
    else:
      model = self.build_model(X_train, Y_train)

    self.model = model

    self.evaluate(X_test, Y_test)

  def evaluate(self, X_test, Y_test):
    preds = self.model.evaluate(X_test, Y_test, batch_size=32, verbose=1, sample_weight=None)
    print()
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))

  def predict(self):

    img = image.load_img(self.img_path, target_size=(64, 64))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    x = x/255
    res = self.model.predict(x)
    K.clear_session()
    res_value = [item for sublist in res for item in sublist][0]

    if res_value > 0.4:
      return "Happy"
    else:
      return "Not happy"


  def build_model(self, X_train, Y_train):
    input_shape = X_train.shape[1:]

    model = Sequential()
    kernel_size = (7, 7)
    model.add(Dense(3, input_shape=input_shape))
    model.add(ZeroPadding2D((3, 3)))

    model.add(Conv2D(12, kernel_size, strides = (1, 1), name = 'conv0'))
    model.add(BatchNormalization(axis = 3, name = 'bn0'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (5, 5), strides = (1, 1), name = 'conv1'))
    model.add(BatchNormalization(axis = 3, name = 'bn1'))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #flatten
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # model.add(Activation('sigmoid'))
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=40, batch_size=12)

    saved_model_path = os.path.join(self.dir_path, 'model.h5')
    model.save(saved_model_path)

    return model

  # def build_model2(self, X_train, Y_train):
  #   input_shape = X_train.shape[1:]

  #   model = Sequential()
  #   kernel_size = (7, 7)
  #   model.add(Dense(32, input_shape=input_shape))
  #   model.add(ZeroPadding2D((3, 3)))
  #   model.add(Conv2D(32, kernel_size, strides = (1, 1), name = 'conv0'))
  #   model.add(BatchNormalization(axis = 3, name = 'bn0'))
  #   model.add(Activation("relu"))
  #   model.add(MaxPooling2D(pool_size=(2, 2)))
  #   model.add(Flatten())
  #   model.add(Dense(1, activation='sigmoid'))
  #   # model.add(Activation('sigmoid'))
  #   model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
  #   model.fit(X_train, Y_train, epochs=40, batch_size=32)
  #   model.save('model.h5')

  #   return model

  def load_dataset(self):
    train_data_set_path = os.path.join(self.dir_path, 'data/train_happy.h5')
    test_data_set_path = os.path.join(self.dir_path, 'data/test_happy.h5')

    train_dataset = h5py.File(train_data_set_path, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File(test_data_set_path, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes