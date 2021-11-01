from typing import List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers.experimental.preprocessing import CenterCrop


class UNetBlockDown(keras.layers.Layer):
    def __init__(self, n_filters: int):
        super(UNetBlockDown, self).__init__()
        self.conv1 = keras.layers.Conv2D(n_filters, 3, activation='relu',
                                         kernel_initializer=keras.initializers.he_normal())
        self.conv2 = keras.layers.Conv2D(n_filters, 3, activation='relu',
                                         kernel_initializer=keras.initializers.he_normal())
        self.max_pooling = keras.layers.MaxPool2D(strides=2)
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x, self.max_pooling(x)


class UNetBlockUp(keras.layers.Layer):
    def __init__(self, in_channels: int):
        super(UNetBlockUp, self).__init__()
        self.up_conv = keras.layers.Conv2DTranspose(in_channels//2, 3, strides=2, padding='same',
                                                    kernel_initializer=keras.initializers.he_normal())
        self.conv1 = keras.layers.Conv2D(in_channels//2, 3, activation='relu',
                                         kernel_initializer=keras.initializers.he_normal())
        self.conv2 = keras.layers.Conv2D(in_channels//2, 3, activation='relu',
                                         kernel_initializer=keras.initializers.he_normal())
    
    def call(self, inputs, skipped):
        # print(inputs.shape, skipped.shape)
        x = self.up_conv(inputs)
        # print(x.shape)
        x = tf.concat((skipped, x), axis=-1)
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
        self.BN = keras.layers.BatchNormalization()

        curr_image_size = self.image_size
        filters = n_filters
        for level in range(depth-1):
            filters = n_filters * (2**level)
            self.contracting_path.append(UNetBlockDown(filters))
            print(f'Level: {level}, # filters = {filters}')
            curr_image_size = curr_image_size - [4, 4]
            print(f'Img size after 2 conv: {curr_image_size}')
            curr_image_size = curr_image_size // 2
            print(f'Img size after max pooling: {curr_image_size}')
        print('End of contracting path')
        self.bottom_conv1 = keras.layers.Conv2D(filters=filters, kernel_size=3, activation='relu',
                                                kernel_initializer=keras.initializers.he_normal())
        self.bottom_conv2 = keras.layers.Conv2D(filters=filters, kernel_size=3, activation='relu',
                                                kernel_initializer=keras.initializers.he_normal())
        curr_image_size = curr_image_size - [4, 4]
        print(f'Img size after bottom convs: {curr_image_size}')
        print('Begin expanding path')
        for level in range(depth-1):
            self.expanding_path.append(UNetBlockUp(filters))
            filters = filters // 2
            print(f'with {filters} filters')
            curr_image_size = 2 * curr_image_size
            print(f'Img size after transp. conv: {curr_image_size}')
            self.cropping_layers.append(CenterCrop(height=curr_image_size[1], width=curr_image_size[0]))
            curr_image_size = curr_image_size - [4, 4]
            print(f'Img size after 2 convs: {curr_image_size}')
        self.output_size = curr_image_size
        self.classifier = keras.layers.Conv2D(1, 1, padding='same', kernel_initializer=keras.initializers.he_normal())
    
    def call(self, inputs):
        skipped = []
        x = self.BN(inputs)
        for i, layer in enumerate(self.contracting_path):
            skip, x = layer(x)
            skipped.append(skip)
        skipped = reversed(skipped)
        x = self.bottom_conv2(self.bottom_conv1(x))
        for skip, crop, up_layer in zip(skipped, self.cropping_layers, self.expanding_path):
            cropped = crop(skip)
            # print(skip.shape, cropped.shape, x.shape)
            x = up_layer(x, cropped)
        return self.classifier(x)

    def print_info(self):
        print(f"Depth of network is {len(self.contracting_path)}")
        print(f"Input image size is {self.image_size}")
        print("Cropped inputs:")
        for i, clayer in enumerate(self.cropping_layers):
            print(f"{i+1}-th: w: {clayer.target_width}, h: {clayer.target_height}")
