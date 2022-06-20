import tensorflow as tf
import typing
from tensorflow import keras


projection_dim = 512


def enblock(x):
    bn = keras.layers.BatchNormalization()(x)
    relu = keras.layers.ReLU()(bn)
    conv = keras.layers.Conv2D(filters=1, kernel_size=3, padding="same")(relu)

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


def create(image_size: typing.Tuple[int, int]):
    input = keras.layers.Input(shape=image_size)  # (128, 128, 1)

    x = keras.layers.Conv2D(filters=4, kernel_size=3, padding="same")(input)  # (128, 128, 4)
    x = keras.layers.Dropout()(x)  # (128, 128, 4)

    en1 = enblock(x)  # (128, 128, 4)

    x = keras.layers.Conv2D(filters=2, kernel_size=3, strides=2, padding="same")(en1)  # (64, 64, 4)

    en2 = enblock(enblock(x))  # (64, 64, 4)

    x = keras.layers.Conv2D(filters=2, kernel_size=3, strides=2, padding="same")(en2)  # (32, 32, 8)

    en3 = enblock(enblock(x))  # (32, 32, 8)

    x = keras.layers.Conv2D(filters=2, kernel_size=3, strides=2, padding="same")(en3)  # (16, 16, 16)

    en4 = enblock(enblock(x))  # (16, 16, 16)

    lin_proj = keras.layers.Conv2D(filters=4, kernel_size=3, padding="same")(en4)  # (16, 16, 64)
    lin_proj = keras.layers.Permute((-1, 1))(lin_proj)  # (64, 16, 16)
    lin_proj = keras.layers.Reshape((64, 256))(lin_proj)  # (64, 256)

    x = transformer_block(lin_proj)
    x = transformer_block(x)
    x = transformer_block(x)
    x = transformer_block(x)  # (64, 256)

    x = keras.layers.Permute((-1, 1))(x)  # (256, 64)
    x = keras.layers.Reshape((16, 16, -1))(x)  # (16, 16, 64)

    feat_map = keras.layers.Conv3D(filters=1, kernel_size=3, strides=(1, 1, 2), padding="same")(x)  # (16, 16, 32)
    feat_map = keras.layers.Conv3D(filters=1, kernel_size=3, strides=(1, 1, 2), padding="same")(feat_map)  # (16, 16, 16)

    x = keras.layers.Concatenate(axis=-1)([feat_map, en4])  # (16, 16, 32)
    x = enblock(enblock(x))  # (16, 16, 32)

    x = keras.layers.Conv2D(filters=1, kernel_size=3, padding="same")(x)  # (16, 16, 32)
    x = keras.layers.Conv2DTranspose(filters=1, kernel_size=3, padding="same", strides=2)(x)  # (32, 32, 32)
    x = keras.layers.Conv3D(filters=1, kernel_size=3, strides=(1, 1, 4), padding="same")(x)  # (32, 32, 8)

    x = keras.layers.Concatenate(axis=-1)([x, en3])  # (32, 32, 16)
    x = enblock(enblock(x))  # (32, 32, 16)

    x = keras.layers.Conv2D(filters=1, kernel_size=3, padding="same")(x)  # (32, 32, 16)
    x = keras.layers.Conv2DTranspose(filters=1, kernel_size=3, padding="same", strides=2)(x)  # (64, 64, 16)
    x = keras.layers.Conv3D(filters=1, kernel_size=3, strides=(1, 1, 4), padding="same")(x)  # (64, 64, 4)

    x = keras.layers.Cocnatenate(axis=-1)([x, en2])
    x = enblock(enblock(x))  # (64, 64, 8)

    x = keras.layers.Conv2D(filters=1, kernel_size=3, padding="same")(x)  # (64, 64, 8)
    x = keras.layers.Conv2DTranspose(filters=1, kernel_size=3, padding="same", strides=2)(x)  # (128, 128, 4)
    x = keras.layers.Conv3D(filters=1, kernel_size=3, strides=(1, 1, 4), padding="same")(x)  # (128, 128, 1)

    x = keras.layers.Concatenate(axis=-1)([x, en1])  # (128, 128, 2)
    x = enblock(enblock(x))  # (128, 128, 2)

    out = keras.layers.Conv3D(filters=1, kernel_size=(1, 1, 3), padding="same", strides=(1, 1, 2),
                              activation="softmax")(x)  # (128, 128, 1)

    return keras.Model(inputs=input, outputs=out)
