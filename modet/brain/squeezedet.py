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

import keras
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Dense, Conv2D, Conv1D, MaxPool2D, GlobalAveragePooling2D, concatenate, Input, Reshape, Lambda, concatenate
from keras.initializers import TruncatedNormal
from tensorflow.python.client import device_lib
from keras.utils import multi_gpu_model

class SqueezeDet(object):
    """
    The Biggy
    """

    def __init__(self, optimizer="Adam", multi=False, save="SqueezeDet.{epoch:02d}.h5"):
        self.save = save
        self.model = self.__build(multi=multi)
        self.model.compile(optimizer, self.__SDetLoss)

    def __SDetLoss(self,yTrue,yPred):
        return 1e-7*(K.sum(K.square((yPred-yTrue))))

    def __build(self, multi=False):

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

        # conv31 = Conv2D(filters=627*(4+1), kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu", kernel_initializer=TruncatedNormal(stddev=1e-11))(f12)
        # Shapes it to self.config.ANCHOR_PER_GRID * (self.config.CLASSES + 1 + 4)

        output_flat = GlobalAveragePooling2D()(f12)
        output_expanded = Lambda(lambda x: K.expand_dims(x, axis=-1))(output_flat)

        # Final shaping convolution
        conv31 = Conv1D(filters=627, kernel_size=5, strides=1, padding="SAME", activation="relu", kernel_initializer=TruncatedNormal(stddev=1e-11))(output_expanded)
        
        # Output Collection
        outputs = []
        for i in range(627):
            anchor = Lambda(lambda x : x[:,:,i])(conv31)
            # net_bounds = Dense(32, activation="relu", kernel_initializer=TruncatedNormal(stddev=1e-11))(anchor)
            # net_bounds = Dense(16, activation="relu", kernel_initializer=TruncatedNormal(stddev=1e-11))(net_bounds)
            # net_bounds = Dense(4, activation="relu", kernel_initializer=TruncatedNormal(stddev=1e-11))(net_bounds)
            net_bounds = Dense(4, activation="relu", kernel_initializer=TruncatedNormal(stddev=1e-11))(anchor)
            net_confidence = Dense(1, activation="sigmoid", kernel_initializer=TruncatedNormal(stddev=1e-11))(net_bounds)
            outputs.append(concatenate([net_bounds, net_confidence]))

        # And, of course, the model
        model = Model(inputs=in_layer, outputs=outputs)

        if multi:
            local_device_protos = device_lib.list_local_devices()
            gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
            num_gpu = len(gpus)
            model = multi_gpu_model(model, gpus=num_gpu)

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
        self.model.fit_generator(manager, epochs=epochs, shuffle=True, callbacks=[keras.callbacks.ModelCheckpoint(self.save, monitor='val_loss')]) 
