# Copyright (c) 2019 Team MODAP
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPool2D, GlobalAveragePooling2D, concatenate, Input, Reshape
from keras.initializers import TruncatedNormal

class SqueezeDet(object):
    """
    The Biggy
    """

    def __init__(self, optimizer="Adam", loss="mae"):
        self.model = self.__build()
        self.model.compile(optimizer, loss)

    def __build(self):

        # the input
        in_layer = Input(batch_shape=(None, 1280, 720, 3), name="input")

        # Firstly, extra the features
        conv0 = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME", activation="relu", kernel_initializer=TruncatedNormal(stddev=1e-11))(in_layer) 
        pool0 = MaxPool2D(pool_size=(3,3), padding="SAME", strides=(2, 2))(conv0) 

        # ok, 黑喂狗.... Begin squeezin' 
        f0 = self.__fire(pool0, 16, 32, 32)
        f1 = self.__fire(f0, 16, 32, 32)
        
        # Max Pool
        pool1 = MaxPool2D(pool_size=(3, 3), padding="SAME", strides=(2, 2))(f1)

        # Aaaaaand, we continue to squeeze it
        f2 = self.__fire(pool1, 32, 64, 64)
        f3 = self.__fire(f2, 32, 64, 64)
        f4 = self.__fire(f3, 32, 64, 64)
        f5 = self.__fire(f4, 32, 64, 64)

        # Max Pool again
        pool2 = MaxPool2D(pool_size=(3, 3), padding="SAME", strides=(2, 2))(f5)

        # Más squeezy
        f6 = self.__fire(pool2, 64, 128, 128)
        f7 = self.__fire(f6, 64, 128, 128)
        f8 = self.__fire(f7, 64, 128, 128)
        f9 = self.__fire(f8, 64, 128, 128)

        # Max Pool Still
        pool3 = MaxPool2D(pool_size=(3, 3), padding="SAME", strides=(2, 2))(f9)

        # Más squeezy
        f10 = self.__fire(pool3, 128, 192, 128)
        f11 = self.__fire(f10, 128, 192, 128)
        f12 = self.__fire(f11, 128, 192, 128)

        # Final shaping convolution
        conv31 = Conv2D(filters=627*(4+1), kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu", kernel_initializer=TruncatedNormal(stddev=1e-11))(f12)
        # Shapes it to self.config.ANCHOR_PER_GRID * (self.config.CLASSES + 1 + 4)

        yHat = GlobalAveragePooling2D()(conv31)
        # And, of course, the model
        model = Model(inputs=in_layer, outputs=yHat)

        return model

    def __fire(self, in_layer, squeeze_filters, expand_filters_small, expand_filters_large):
        """__fire
        A SqueezeNet Fire Module
        (If you don't know what this is you *need* to look it up)

        :param in_layer: layer input
        :param squeeze_filters: the number of filters for the squeeze layer
        :param expand_filters_small: the number of filters for the 1x1 expand layer
        :param expand_filter_large: the number of filters for the 3x3
        """
        squeeze = Conv2D(filters=squeeze_filters, kernel_size=(1, 1), use_bias=True, activation="relu", kernel_initializer=TruncatedNormal(stddev=1e-11))(in_layer)
        smallExpand = Conv2D(filters=expand_filters_small, kernel_size=(1, 1), use_bias=True, activation="relu", kernel_initializer=TruncatedNormal(stddev=1e-11))(squeeze)
        largeExpand = Conv2D(filters=expand_filters_large, kernel_size=(3, 3), use_bias=True, padding="SAME", activation="relu", kernel_initializer=TruncatedNormal(stddev=1e-11))(squeeze)
        return concatenate([smallExpand, largeExpand], axis=3)

    def fit(self, manager, epochs=10):
        self.model.fit_generator(manager, epochs=epochs, shuffle=True) 
