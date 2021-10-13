import tensorflow as tf
from tensorflow import keras

class UNetBlockDown(keras.layers.Layer):
    def __init__(self, n_filters: int, max_pooling: bool=True):
        super(UNetBlockDown, self).__init__()
        self.do_pooling = max_pooling
        self.conv1 = keras.layers.Conv2D(n_filters, 3, activation='relu')
        self.conv2 = keras.layers.Conv2D(n_filters, 3, activation='relu')
        self.max_pooling = keras.layers.MaxPool2D(strides=2)
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        if self.max_pooling:
            return self.max_pooling(x)
        return x

class UNetBlockUp(keras.layers.Layer):
    def __init__(self, in_channels: int):
        self.up_conv = keras.layers.Conv2DTranspose(in_channels//2, 3, strides=2, padding='same')
        self.conv1 = keras.layers.Conv2D(in_channels//2, 3, activation='relu')
        self.conv2 = keras.layers.Conv2D(in_channels//2, 3, activation='relu')
    
    def call(self, inputs):
        x = self.up_conv(inputs)
        x = self.conv1(x)
        x = self.conv2(x)

class UNet(keras.models.Model):
