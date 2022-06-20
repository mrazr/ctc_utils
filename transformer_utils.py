import math

import numpy as np
import tensorflow as tf


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
    def __init__(self, patch_size):
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
