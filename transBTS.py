import tensorflow as tf
import typing
from tensorflow import keras


projection_dim = 256
DATA_FORMAT = "channels_first"


def enblock(x, chans):
    global DATA_FORMAT
    bn = keras.layers.BatchNormalization()(x)
    relu = keras.layers.ReLU()(bn)
    conv = keras.layers.Conv2D(data_format=DATA_FORMAT, filters=chans, kernel_size=3, padding="same")(relu)

    return conv


# def deblock(x):
#     bn = keras.layers.BatchNormalization()(x)
#     relu = keras.layers.ReLU()(bn)
#
#


def transformer_block(input):
    x = keras.layers.LayerNormalization()(input)
    attention_output = keras.layers.MultiHeadAttention(num_heads=3, key_dim=projection_dim)(x, x)

    added = keras.layers.Add()([input, attention_output])

    normed = keras.layers.LayerNormalization()(added)

    x = mlp(normed, [projection_dim * 2, projection_dim], 0.1)

    return keras.layers.Add()([x, added])


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = keras.layers.Dense(units=units, activation='gelu')(x)
        x = keras.layers.Dropout(dropout_rate)(x)

    return x


def create(image_size: typing.Tuple[int, int], data_format: str = 'channels_last'):
    input = keras.layers.Input(shape=image_size)  # (128, 128, 1) if `data_format` == 'channels_last'

    x = keras.layers.Conv2D(data_format=data_format, filters=4, kernel_size=3, padding="same")(input)  # (128, 128, 4)
    x = keras.layers.Dropout(0.1)(x)  # (128, 128, 4)

    en1 = enblock(x, 4)  # (128, 128, 4)

    x = keras.layers.Conv2D(data_format=data_format, filters=8, kernel_size=3, strides=(2, 2), padding="same")(en1)  # (64, 64, 8)

    en2 = enblock(enblock(x, 8), 8)  # (64, 64, 8)

    x = keras.layers.Conv2D(data_format=data_format, filters=16, kernel_size=3, strides=(2, 2), padding="same")(en2)  # (32, 32, 16)

    en3 = enblock(enblock(x, 16), 16)  # (32, 32, 16)

    x = keras.layers.Conv2D(data_format=data_format, filters=32, kernel_size=3, strides=(2, 2), padding="same")(en3)  # (16, 16, 32)

    en4 = enblock(enblock(x, 32), 32)  # (16, 16, 32)

    lin_proj = keras.layers.Conv2D(data_format=data_format, filters=128, kernel_size=3, padding="same")(en4)  # (16, 16, 128)
    if data_format == "channels_last":
        lin_proj = keras.layers.Permute((3, 1, 2))(lin_proj)  # (128, 16, 16)

    lin_proj = keras.layers.Reshape((128, 256))(lin_proj)  # (128, 256)

    x = transformer_block(lin_proj)
    x = transformer_block(x)
    x = transformer_block(x)
    x = transformer_block(x)  # (128, 256)

    if data_format == "channels_last":
        x = keras.layers.Permute((2, 1))(x)  # (256, 128)
    x = keras.layers.Reshape((16, 16, -1))(x)  # (16, 16, 128)

    feat_map = keras.layers.Conv2D(data_format=data_format, filters=64, kernel_size=1, padding="same")(x)  # (16, 16, 64)
    feat_map = keras.layers.Conv2D(data_format=data_format, filters=32, kernel_size=1, padding="same")(feat_map)  # (16, 16, 32)

    x = keras.layers.Concatenate(axis=-1)([feat_map, en4])  # (16, 16, 64)
    x = enblock(enblock(x, 64), 64)  # (16, 16, 64)

    x = keras.layers.Conv2D(data_format=data_format, filters=64, kernel_size=3, padding="same")(x)  # (16, 16, 64)
    x = keras.layers.Conv2DTranspose(filters=64, kernel_size=3, padding="same", strides=2)(x)  # (32, 32, 64)
    x = keras.layers.Conv2D(data_format=data_format, filters=16, kernel_size=1, padding="same")(x)  # (32, 32, 16)

    x = keras.layers.Concatenate(axis=-1)([x, en3])  # (32, 32, 32)
    x = enblock(x, 32)  # (32, 32, 32)

    x = keras.layers.Conv2D(data_format=data_format, filters=32, kernel_size=3, padding="same")(x)  # (32, 32, 32)
    x = keras.layers.Conv2DTranspose(filters=32, kernel_size=3, padding="same", strides=2)(x)  # (64, 64, 32)
    x = keras.layers.Conv2D(data_format=data_format, filters=8, kernel_size=1, padding="same")(x)  # (64, 64, 8)

    x = keras.layers.Concatenate(axis=-1)([x, en2])  # (64, 64, 16)
    x = enblock(x, 16)  # (64, 64, 16)

    x = keras.layers.Conv2D(data_format=data_format, filters=16, kernel_size=3, padding="same")(x)  # (64, 64, 16)
    x = keras.layers.Conv2DTranspose(filters=16, kernel_size=3, padding="same", strides=2)(x)  # (128, 128, 16)
    x = keras.layers.Conv2D(data_format=data_format, filters=4, kernel_size=1, padding="same")(x)  # (128, 128, 4)

    x = keras.layers.Concatenate(axis=-1)([x, en1])  # (128, 128, 8)
    x = enblock(x, 8)  # (128, 128, 8)

    out = keras.layers.Conv2D(data_format=data_format, filters=1, kernel_size=1, padding="same",
                              activation="softmax")(x)  # (128, 128, 1)

    return keras.Model(inputs=input, outputs=out)
