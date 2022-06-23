import tensorflow as tf
import typing
from tensorflow import keras


projection_dim = 256
DATA_FORMAT = "channels_last"


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
    global projection_dim
    projection_dim = (image_size[0] // 8) * (image_size[1] // 8)
    input = keras.layers.Input(shape=image_size)  # (w, h, 1) if `data_format` == 'channels_last'

    x = keras.layers.Conv2D(data_format=data_format, filters=4, kernel_size=3, padding="same")(input)  # (w, h, 4)
    x = keras.layers.Dropout(0.1)(x)  # (w, h, 4)

    en1 = enblock(x, 4)  # (w, h, 4)

    x = keras.layers.Conv2D(data_format=data_format, filters=8, kernel_size=3, strides=(2, 2), padding="same")(en1)  # (w/2, h/2, 8)

    en2 = enblock(enblock(x, 8), 8)  # (w/2, h/2, 8)

    x = keras.layers.Conv2D(data_format=data_format, filters=16, kernel_size=3, strides=(2, 2), padding="same")(en2)  # (w/4, h/4, 16)

    en3 = enblock(enblock(x, 16), 16)  # (w/4, h/4, 16)

    x = keras.layers.Conv2D(data_format=data_format, filters=32, kernel_size=3, strides=(2, 2), padding="same")(en3)  # (w/8, h/8, 32)

    en4 = enblock(enblock(x, 32), 32)  # (w/8, h/8, 32)

    lin_proj = keras.layers.Conv2D(data_format=data_format, filters=128, kernel_size=3, padding="same")(en4)  # (w/8, h/8, 128)
    if data_format == "channels_last":
        lin_proj = keras.layers.Permute((3, 1, 2))(lin_proj)  # (128, w/8, h/8)

    lin_proj = keras.layers.Reshape((128, -1))(lin_proj)  # (128, w*h/64)

    x = transformer_block(lin_proj)
    x = transformer_block(x)
    x = transformer_block(x)
    x = transformer_block(x)  # (128, w*h/64)

    if data_format == "channels_last":
        x = keras.layers.Permute((2, 1))(x)  # (w*h/64, 128)
    x = keras.layers.Reshape((image_size[0] // 8, image_size[1] // 8, -1))(x)  # (w/8, h/8, 128)

    feat_map = keras.layers.Conv2D(data_format=data_format, filters=64, kernel_size=1, padding="same")(x)  # (w/8, h/8, 64)
    feat_map = keras.layers.Conv2D(data_format=data_format, filters=32, kernel_size=1, padding="same")(feat_map)  # (w/8, h/8, 32)

    x = keras.layers.Concatenate(axis=-1)([feat_map, en4])  # (w/8, h/8, 64)
    x = enblock(enblock(x, 64), 64)  # (w/8, h/8, 64)

    x = keras.layers.Conv2D(data_format=data_format, filters=64, kernel_size=3, padding="same")(x)  # (w/8, h/8, 64)
    x = keras.layers.Conv2DTranspose(data_format=data_format, filters=64, kernel_size=3, padding="same", strides=2)(x)  # (w/4, h/4, 64)
    x = keras.layers.Conv2D(data_format=data_format, filters=16, kernel_size=1, padding="same")(x)  # (w/4, h/4, 16)

    x = keras.layers.Concatenate(axis=-1)([x, en3])  # (w/4, h/4, 32)
    x = enblock(x, 32)  # (w/4, h/4, 32)

    x = keras.layers.Conv2D(data_format=data_format, filters=32, kernel_size=3, padding="same")(x)  # (w/4, h/4, 32)
    x = keras.layers.Conv2DTranspose(data_format=data_format, filters=32, kernel_size=3, padding="same", strides=2)(x)  # (w/2, h/2, 32)
    x = keras.layers.Conv2D(data_format=data_format, filters=8, kernel_size=1, padding="same")(x)  # (w/2, h/2, 8)

    x = keras.layers.Concatenate(axis=-1)([x, en2])  # (w/2, h/2, 16)
    x = enblock(x, 16)  # (w/2, h/2, 16)

    x = keras.layers.Conv2D(data_format=data_format, filters=16, kernel_size=3, padding="same")(x)  # (w/2, h/2, 16)
    x = keras.layers.Conv2DTranspose(data_format=data_format, filters=16, kernel_size=3, padding="same", strides=2)(x)  # (w, h, 16)
    x = keras.layers.Conv2D(data_format=data_format, filters=4, kernel_size=1, padding="same")(x)  # (w/2, h/2, 4)

    x = keras.layers.Concatenate(axis=-1)([x, en1])  # (w, h, 8)
    x = enblock(x, 8)  # (w, h, 8)

    out = keras.layers.Conv2D(data_format=data_format, filters=1, kernel_size=1, padding="same",
                              activation="sigmoid")(x)  # (w, h, 1)

    return keras.Model(inputs=input, outputs=out)
