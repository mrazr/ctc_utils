import math
import os
import typing

import numpy as np
import tensorflow as tf
from skimage import io
from tensorflow import keras
from tensorflow.keras import layers

import ctc_dataset
from my_ctc_utils import ElasticDeform


class Preprocessing(keras.layers.Layer):
    def __init__(self, resize_to: typing.Tuple[int, int], seed: int = 42, deform: bool = True):
        super(Preprocessing, self).__init__()
        self.img_resize = keras.layers.experimental.preprocessing.Resizing(*resize_to[::-1])
        self.mask_resize = keras.layers.experimental.preprocessing.Resizing(*resize_to[::-1])
        self._rng = tf.random.Generator.from_seed(seed)
        self.elastic_deform = ElasticDeform() if deform else None

    def call(self, inputs):
        img = inputs[0]
        mask = inputs[1]

        flip_seed_h = self._rng.uniform_full_int(shape=[2], dtype=tf.int32)
        flip_seed_v = self._rng.uniform_full_int(shape=[2], dtype=tf.int32)

        # img_ = img / 255.0

        img_ = self.img_resize(img)
        img_ = tf.image.stateless_random_flip_left_right(img_, flip_seed_h)
        img_ = tf.image.stateless_random_flip_up_down(img_, flip_seed_v)
        #img_ = self.img_flip(img_)

        mask_ = self.mask_resize(mask)
        mask_ = np.where(mask_ > 0, 1, 0).astype(np.uint8)
        mask_ = tf.image.stateless_random_flip_left_right(mask_, flip_seed_h)
        mask_ = tf.image.stateless_random_flip_up_down(mask_, flip_seed_v)

        if self.elastic_deform is not None:
            img_ = np.expand_dims(np.squeeze(img_), axis=-1)
            mask_ = np.expand_dims(np.squeeze(mask_), axis=-1)
            img_, mask_ = self.elastic_deform(img_, mask_)
            img_ = np.expand_dims(img_, axis=0)
            mask_ = np.expand_dims(mask_, axis=0)

        return img_, mask_


class PatchTokenizer(tf.keras.layers.Layer):
    def __init__(self, patch_size: int = 4):
        super().__init__(self)
        self.patch_size: int = patch_size
        self.flat = tf.keras.layers.Flatten()

    def __call__(self, inputs):
        img = inputs
        # label = inputs[1]

        n_patches = (img.shape[0] * img.shape[1]) // (self.patch_size * self.patch_size)
        rows = int(math.sqrt(n_patches))

        tokens = np.zeros((n_patches, self.patch_size * self.patch_size))

        # patches = np.zeros((n_patches, self.patch_size, self.patch_size), np.uint8)

        for row in range(int(math.sqrt(n_patches))):
            for col in range(int(math.sqrt(n_patches))):
                patch = img[row * self.patch_size:self.patch_size * (row + 1),
                        col * self.patch_size:self.patch_size * (col + 1)]
                # print(f'patch size is {patch.shape}')
                # patches[row * rows + col] = patch
                # tokens[row * n_patches + col] =
                tokens[row * rows + col] = np.ravel()
        return tf.constant(tokens)


class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size: int):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x


def position_embedding(
        projected_patches, num_patches, projection_dim):
    # Build the positions.
    positions = tf.range(start=0, limit=num_patches, delta=1)

    # Encode the positions with an Embedding layer.
    encoded_positions = tf.keras.layers.Embedding(
        input_dim=num_patches, output_dim=projection_dim
    )(positions)

    # Add encoded positions to the projected patches.
    return projected_patches + encoded_positions


learning_rate = 0.001
weight_decay = 0.0001
batch_size = 1
num_epochs = 20
image_size = 128
patch_size = 4
num_patches = (image_size // patch_size) ** 2
projection_dim = 256
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim
]
transformer_layers = 8
mlp_head_units = [2048, 1024]


def transformer_block(input, output_size: int):
    x1 = layers.LayerNormalization(epsilon=1e-6)(input)  # x1 is (64, 256)
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim,
                                   dropout=0.1)(x1, x1)  # this too should be (64, 256)
    x2 = layers.Add()([attention_output, input])  # (64, 256)

    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)  # (64, 256)

    x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)  # (64, 256)

    return layers.Add()([x3, x2])


def create_vitbis(input_shape: typing.Tuple[int, int]):
    inputs = layers.Input(shape=(image_size, image_size, 1))

    patches = Patches(patch_size)(inputs)  # produces 128 * 128 / (16 * 16) = 64 patches of size 16x16

    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)  # encodes them

    skip_to_decoder = []

    x = encoded_patches  # (64, 256)
    change_to_size = (patch_size * patch_size) // 4  # 256 // 4 = 64
    for i in range(2):
        transformer_output = transformer_block(x, num_patches * patch_size * patch_size)  # (64, 256)
        skip_to_decoder.append(layers.Dense(units=change_to_size)(transformer_output))  # [0](64, 64), [1](64, 32)
        change_to_size = change_to_size // 2
        x = transformer_output  # (64, 256)

    change_to_size *= 2  # 64
    x = layers.Dense(units=256)(x)  # (64, 64)
    # x = layers.Dense(units=change_to_size)(x)  # (64, 64)

    skip_to_decoder = list(reversed(skip_to_decoder))

    for i in range(2):
        # concat = layers.Concatenate()([x, skip_to_decoder[i]])
        # added = layers.Concatenate(axis=0)([x, skip_to_decoder[i]])
        # added = layers.Add()([x, skip_to_decoder[i]])
        upsampled = layers.Dense(units=projection_dim)(skip_to_decoder[i])
        concat = layers.Concatenate(axis=0)([x, upsampled])
        transformer_output = transformer_block(concat, change_to_size)
        change_to_size *= 2
        x = transformer_output   #layers.Dense(units=change_to_size)(transformer_output)

    out = layers.Dense(units=(image_size * image_size) // num_patches)(x)
    out = layers.Reshape((image_size, image_size))(out)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    return model


class VitbisSequence(keras.utils.Sequence):
    def __init__(self, img_paths: typing.List[str], mask_paths: typing.List[str], image_size: typing.Tuple[int, int], deform: bool = True,
                 sample_weights=None):
        super().__init__()
        if sample_weights is None:
            sample_weights = list()
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        # self.center_crop = keras.layers.experimental.preprocessing.CenterCrop(crop_size[0], crop_size[1])
        self.sample_weights = sample_weights
        self.preprocessing = Preprocessing(resize_to=image_size, deform=deform)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img = np.expand_dims(io.imread(self.img_paths[index], as_gray=True).astype(np.float32), axis=(0, -1))
        # print(img.shape, img.dtype)
        mask = np.expand_dims(io.imread(self.mask_paths[index], as_gray=True), axis=(0, -1))
        img, mask = self.preprocessing((img, mask))
        if self.sample_weights is not None:
            return add_sample_weights(img, mask, self.sample_weights)
        return img, mask


def add_sample_weights(image, label, weights: typing.List[float]):
    class_weights = tf.constant(weights)

    sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))

    return image, label, sample_weights


def get_train_val(ds: ctc_dataset.Dataset, img_size: typing.Tuple[int, int], sample_weights: typing.List[float] = None):
    X_train, y_train, X_val, y_val = ds.train_val_split()
    train_seq = VitbisSequence(img_paths=X_train, mask_paths=y_train, image_size=img_size, deform=True,
                               sample_weights=sample_weights)
    val_seq = VitbisSequence(img_paths=X_val, mask_paths=y_val, image_size=img_size, deform=False,
                             sample_weights=sample_weights)

    return train_seq, val_seq