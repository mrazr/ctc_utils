from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import CenterCrop


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
            return x, self.max_pooling(x)
        return x, x


class UNetBlockUp(keras.layers.Layer):
    def __init__(self, in_channels: int):
        self.up_conv = keras.layers.Conv2DTranspose(in_channels//2, 3, strides=2, padding='same')
        self.conv1 = keras.layers.Conv2D(in_channels//2, 3, activation='relu')
        self.conv2 = keras.layers.Conv2D(in_channels//2, 3, activation='relu')
    
    def call(self, inputs, skipped):
        x = self.up_conv(inputs)
        x = tf.concat((skipped, x), axis=0)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNet(keras.models.Model):
    def __init__(self, depth: int, n_filters: int, image_size: Tuple[int, int]):
        super(UNet, self).__init__()
        self.contracting_path: List[UNetBlockDown] = []
        self.expanding_path: List[UNetBlockUp] = []
        self.cropping_layers: List[CenterCrop] = []
        self.crop_sizes: List[Tuple[int, int]] = []
        self.image_size: np.ndarray = np.array(image_size)

        curr_image_size = self.image_size

        for level in range(1, depth + 1):
            filters = n_filters * (2**(level - 1))
            self.contracting_path.append(UNetBlockDown(filters, level != depth))
            curr_image_size = curr_image_size - [4, 4]
            if level != depth:
                curr_image_size = curr_image_size // 2
        
        for level in range(depth-1):
            self.expanding_path.append(UNetBlockUp(filters))
            curr_image_size = 2 * curr_image_size
            self.cropping_layers.append(CenterCrop(height=curr_image_size[1], width=curr_image_size[0]))
            curr_image_size = curr_image_size - [4, 4]
        self.classifier = keras.layers.Conv2D(2, 1, padding='same')
    
    def call(self, inputs):
        skipped = []
        x = inputs
        for i, layer in enumerate(self.contracting_path):
            skip, x = layer(x)
            if i < len(self.contracting_path) - 1:
                skipped.append(skip)

        for skip, crop, up_layer in zip(skipped, self.cropping_layers, self.expanding_path):
            cropped = crop(skip)
            x = up_layer(x, cropped)
        return self.classifier(x)
    
    def print_info(self):
        print(f"Depth of network is {len(self.contracting_path)}")
        print(f"Input image size is {self.image_size}")
        print("Cropped inputs:")
        for i, clayer in enumerate(self.cropping_layers):
            print(f"{i+1}-th: w: {clayer.target_width}, h: {clayer.target_height}")
        