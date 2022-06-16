import os
from collections import namedtuple
from typing import List, Tuple
import random

import numpy as np
from skimage import io
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers.experimental.preprocessing import CenterCrop

from ctc_dataset import Dataset, ST, GT
from img_utils import generate_displacements, deform_image


class UNetDataInfo:
    def __init__(self, input_size: Tuple[int, int], unet_depth: int, padding: str='valid', binary: bool=True):
        self.original_input_size = np.array(input_size)
        self.depth = unet_depth
        self.input_size = 2 * (compute_correct_size(self.original_input_size[0], self.depth, padding=padding),)
        self.cropped_mask_size = 2 * (get_output_map_size(self.input_size[0], self.depth, padding=padding),)
        self.binary: bool = binary


class CTCSequence(keras.utils.Sequence):
    def __init__(self, img_paths: List[str], mask_paths: List[str], unet_info: UNetDataInfo, deform: bool = True,
                 sample_weights=None):
        super(CTCSequence, self).__init__()
        if sample_weights is None:
            sample_weights = list()
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        # self.center_crop = keras.layers.experimental.preprocessing.CenterCrop(crop_size[0], crop_size[1])
        self.sample_weights = sample_weights
        self.preprocessing = Preprocessing(unet_info, deform=deform)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img = np.expand_dims(io.imread(self.img_paths[index], as_gray=True).astype(np.float32), axis=(0, -1))
        # print(img.shape, img.dtype)
        mask = np.expand_dims(io.imread(self.mask_paths[index], as_gray=True), axis=(0, -1))
        img, mask = self.preprocessing((img, mask))
        if self.sample_weights is not None:
            return img, mask, add_sample_weights(img, mask, self.sample_weights)
        return img, mask


class Preprocessing(keras.layers.Layer):
    def __init__(self, unet_info: UNetDataInfo, seed: int = 42, deform: bool = True):
        super(Preprocessing, self).__init__()
        self.unet_info = unet_info
        self.mask_size = unet_info.cropped_mask_size
        self.mask_crop = keras.layers.experimental.preprocessing.CenterCrop(self.mask_size[0], self.mask_size[1])
        self.img_resize = keras.layers.experimental.preprocessing.Resizing(*unet_info.original_input_size)
        self.mask_resize = keras.layers.experimental.preprocessing.Resizing(*unet_info.original_input_size,
                                                                            interpolation='nearest')
        self._rng = tf.random.Generator.from_seed(seed)
        #self.img_flip = keras.layers.experimental.preprocessing.RandomFlip(seed=seed)
        #self.mask_flip = keras.layers.experimental.preprocessing.RandomFlip(seed=seed)
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
        #mask_ = self.mask_flip(mask_)
        # print(mask_.dtype)
        # mask_ = tf.cast(mask_, tf.uint16)
        if self.unet_info.binary:
            mask_ = np.where(mask_ > 0, 1, 0).astype(np.uint8)
        mask_ = tf.image.stateless_random_flip_left_right(mask_, flip_seed_h)
        mask_ = tf.image.stateless_random_flip_up_down(mask_, flip_seed_v)

        if self.elastic_deform is not None:
            img_ = np.expand_dims(np.squeeze(img_), axis=-1)
            mask_ = np.expand_dims(np.squeeze(mask_), axis=-1)
            img_, mask_ = self.elastic_deform(img_, mask_)
            img_ = np.expand_dims(img_, axis=0)
            mask_ = np.expand_dims(mask_, axis=0)

        pad = (self.unet_info.input_size[0] - img_.shape[1]) // 2
        img_ = np.pad(img_, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='reflect')

        # return add_sample_weights(img_, self.mask_crop(mask_))

        return img_, self.mask_crop(mask_)


def add_sample_weights(image, label, weights: List[float]):
    class_weights = tf.constant(weights)

    sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))

    return image, label, sample_weights


class ElasticDeform(keras.layers.Layer):
    def __init__(self, p: float = 0.5, seed: int = 42):
        super(ElasticDeform, self).__init__()
        self.p = p
        self.rng = np.random.default_rng(seed=seed)

    def call(self, img, mask):
        if self.rng.random() > self.p:
            shape = mask.shape if mask.ndim == 2 else mask.shape[:2]
            dx, dy = generate_displacements(shape, 10, 4.5, 8)
            warped_img = deform_image(img, dx, dy, order=1)
            warped_mask = deform_image(mask, dx, dy, order=0)
            return warped_img, warped_mask
        return img, mask


def compute_last_img_size(sz, depth, padding: str='valid') -> int:
    d = 4 if padding == 'valid' else 0
    cs = sz
    for i in range(depth - 1):
        cs -= d
        if cs % 2:
            return 5 # 5 is an odd number (I think)
        cs = cs // 2
    return cs - d


def compute_correct_size(sz, depth, padding: str='valid') -> int:
    pad = 2 * 4 * (depth - 1) + 2 * (depth - 1) + 4 if padding == 'valid' else 0
    cs = sz + pad
    while compute_last_img_size(cs, depth, padding=padding) % 2:
        cs += 2
    return cs


def get_crop_sizes(sz, depth, padding: str='valid') -> List[Tuple[int, int]]:
    sizes: List[Tuple[int, int]] = []
    curr_img_size = compute_last_img_size(sz, depth, padding=padding)
    d = 4 if padding == 'valid' else 0
    for i in range(depth - 1):
        curr_img_size *= 2
        sizes.append((curr_img_size, curr_img_size))
        curr_img_size -= d
    return sizes


def get_output_map_size(sz, depth, padding: str='valid') -> int:
    d = 4 if padding == 'valid' else 0
    return get_crop_sizes(sz, depth, padding=padding)[-1][0] - d


def sz2str(sz) -> str:
    return f'{sz[0]}_{sz[1]}'


def build_unet(input_size: Tuple[int, int] = (256, 256), depth: int = 4, n_filters: int = 32, padding: str='valid', binary: bool=True) -> Tuple[
    UNetDataInfo, keras.Model]:

    size_change = np.array([4, 4]) if padding == 'valid' else np.array([0, 0])

    info = UNetDataInfo(input_size, depth, padding=padding, binary=binary)

    input_tensor = keras.layers.Input(shape=info.input_size + (1,), name='input')
    x = keras.layers.BatchNormalization()(input_tensor)

    do_cropping = padding == 'valid'
    if do_cropping:
        crop_sizes: List[Tuple[int, int]] = get_crop_sizes(info.input_size[0], depth, padding=padding)[::-1]
    else:
        crop_sizes = []
    skip_layers = []

    curr_image_size = np.array(info.input_size)
    filters = n_filters
    for level in range(depth - 1):
        filters = n_filters * (2 ** level)
        layer_name = f'CC_{sz2str(curr_image_size-2)}x{filters}'
        x = keras.layers.Conv2D(filters, 3, activation='relu',
                                kernel_initializer='he_normal', padding=padding, name=layer_name)(x)
        layer_name = f'CC_{sz2str(curr_image_size-size_change)}x{filters}'
        x = keras.layers.Conv2D(filters, 3, activation='relu',
                                kernel_initializer='he_normal', padding=padding, name=layer_name)(x)
        if do_cropping:
            layer_name = f'Skip_{sz2str(crop_sizes[level])}'
            skip_layers.append(keras.layers.experimental.preprocessing.CenterCrop(*crop_sizes[level], name=layer_name)(x))
        else:
            skip_layers.append(x)
        curr_image_size = curr_image_size - size_change
        curr_image_size = curr_image_size // 2
        x = keras.layers.MaxPooling2D(strides=2, name=f'Pool_{sz2str(curr_image_size)}x{filters}')(x)

    x = keras.layers.Conv2D(filters=filters, kernel_size=3, activation='relu',
                            kernel_initializer=keras.initializers.he_normal(), padding=padding,
                            name=f'BC_{sz2str(curr_image_size-size_change//2)}x{filters}')(x)
    x = keras.layers.Conv2D(filters=filters, kernel_size=3, activation='relu',
                            kernel_initializer=keras.initializers.he_normal(), padding=padding,
                            name=f'BC_{sz2str(curr_image_size-size_change)}x{filters}')(x)

    curr_image_size = curr_image_size - size_change
    skipped = skip_layers[::-1]

    for level in range(depth - 1):
        filters = filters // 2
        curr_image_size *= 2
        layer_name = f'E_Up_{sz2str(curr_image_size)}'
        #x = keras.layers.UpSampling2D(size=(2, 2), name=layer_name)(x)
        x = keras.layers.Conv2DTranspose(filters, 3, strides=2, padding='same',
                                         kernel_initializer=keras.initializers.he_normal(), activation='relu',
                                         name=layer_name)(x)
        x = keras.layers.Concatenate(axis=-1, name=f'Expand_{depth-level}_Concat')([x, skipped[level]])
        x = keras.layers.Conv2D(filters, 3, activation='relu',
                                kernel_initializer=keras.initializers.he_normal(), padding=padding,
                                name=f'EC_{sz2str(curr_image_size-size_change//2)}')(x)
        x = keras.layers.Conv2D(filters, 3, activation='relu',
                                kernel_initializer=keras.initializers.he_normal(), padding=padding,
                                name=f'EC_{sz2str(curr_image_size-size_change)}')(x)
        curr_image_size -= size_change
    output = keras.layers.Conv2D(1 if binary else 3, 1, padding='same', kernel_initializer=keras.initializers.he_normal(),
                                 name=f'Output_{sz2str(curr_image_size)}')(x)

    return info, keras.models.Model(inputs=input_tensor, outputs=output)


def get_data(ds: Dataset, unet_info: UNetDataInfo, seed: int = 42, truth: str = 'ST'):
    image_names = []
    ann_names = []

    assert truth in ['GT', 'ST']

    for seq in ds.sequences:
        truths = seq.silver_truths if truth == ST else seq.gold_truths
        seq_folder = os.path.join(seq.root_folder, seq.folder)
        ann_folder = os.path.join(seq.root_folder, seq.folder + '_' + truth)
        image_names.extend([os.path.join(seq_folder, ex[0]) for ex in truths])
        ann_names.extend([os.path.join(ann_folder, ex[1]) for ex in truths])

    return CTCSequence(image_names, ann_names, unet_info, deform=truth == ST)


def get_train_val_data(ds: Dataset, unet_info: UNetDataInfo, seed: int=42, split: float=0.3, sample_weights=None):
    if sample_weights is None:
        sample_weights = list()
    image_names = []
    ann_names = []

    for seq in ds.sequences:
        truths = seq.silver_truths
        seq_folder = os.path.join(seq.root_folder, seq.folder)
        ann_folder = os.path.join(seq.root_folder, seq.folder + '_' + ST)
        image_names.extend([os.path.join(seq_folder, ex[0]) for ex in truths])
        ann_names.extend([os.path.join(ann_folder, ex[1]) for ex in truths])

    rng = np.random.default_rng()
    perm = rng.permutation(len(image_names))
    image_names = np.array(image_names)[perm]
    ann_names = np.array(ann_names)[perm]

    split_idx = int(round(split * len(image_names)))

    val = CTCSequence(image_names[:split_idx], ann_names[:split_idx], unet_info, deform=False,
                      sample_weights=sample_weights)
    train = CTCSequence(image_names[split_idx:], ann_names[split_idx:], unet_info, deform=True,
                        sample_weights=sample_weights)

    return train, val


TrainValInfo = namedtuple('TrainValInfo', ['train_data', 'val_data', 'info'])


def train_unet2(dataset_path: str, image_size: Tuple[int, int], epochs: int=15, unet_depth: int=4, unet_filters:int=32, padding: str='valid', binary: bool=True) -> Tuple[keras.Model, keras.callbacks.History, TrainValInfo]:
    dataset = Dataset(dataset_path)
    info, model = build_unet(image_size, depth=unet_depth, n_filters=unet_filters, padding=padding, binary=binary)

    train_data, val_data = get_train_val_data(dataset, info)

    callbacks = [
        keras.callbacks.ModelCheckpoint(f'./{os.path.basename(dataset_path)}_unet_.hdf', save_best_only=True),
    ]
    loss = keras.losses.BinaryCrossentropy(from_logits=True) if binary else keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-2),
                  loss=loss,
                  metrics=['accuracy'])

    hist = model.fit(train_data, epochs=epochs, validation_data=val_data, callbacks=callbacks)

    return model, hist, TrainValInfo(train_data, val_data, info)
