import os
from pathlib import PurePath
from typing import Union, List, Tuple, Optional
import re
import random
from collections import namedtuple

from skimage import io, color
from skimage.transform import resize
from skimage import measure
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as nd
import tensorflow as tf
import tensorflow.keras as keras

import unet

GT = "GT"
ST = "ST"
SEG = "SEG"
TRA = "TRA"
RES = "RES"

man_track_re = re.compile('man_track(\d{3,4}).tif')
man_seg_re = re.compile('man_seg(\d{3,4}).tif')


class UNetDataInfo:
    def __init__(self, input_size: Tuple[int, int], unet_depth: int):
        self.original_input_size = np.array(input_size)
        self.depth = unet_depth
        self.input_size = 2 * (compute_correct_size(self.original_input_size[0], self.depth),)
        self.cropped_mask_size = 2 * (get_output_map_size(self.input_size[0], self.depth),)


class CTCSequence(keras.utils.Sequence):
    def __init__(self, img_paths: List[str], mask_paths: List[str], unet_info: UNetDataInfo, deform: bool=True):
        super(CTCSequence, self).__init__()
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        # self.center_crop = keras.layers.experimental.preprocessing.CenterCrop(crop_size[0], crop_size[1])
        self.preprocessing = Preprocessing(unet_info, deform=deform)
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        # print(self.img_paths[index], self.mask_paths[index])
        img = np.expand_dims(io.imread(self.img_paths[index], as_gray=True).astype(np.float32), axis=-1)
        # print(img.shape, img.dtype)
        mask = np.expand_dims(io.imread(self.mask_paths[index], as_gray=True), axis=-1)
        img, mask = self.preprocessing((img, mask))
        return np.array([img]), np.array([mask])


class Preprocessing(keras.layers.Layer):
    def __init__(self, unet_info: UNetDataInfo, seed: int=42, deform: bool=True):
        super(Preprocessing, self).__init__()
        self.unet_info = unet_info
        self.mask_size = unet_info.cropped_mask_size
        self.mask_crop = keras.layers.experimental.preprocessing.CenterCrop(self.mask_size[0], self.mask_size[1])
        self.img_resize = keras.layers.experimental.preprocessing.Resizing(*unet_info.original_input_size)
        self.mask_resize = keras.layers.experimental.preprocessing.Resizing(*unet_info.original_input_size,
                                                                            interpolation='nearest')
        self.img_flip = keras.layers.experimental.preprocessing.RandomFlip(seed=seed)
        self.mask_flip = keras.layers.experimental.preprocessing.RandomFlip(seed=seed)
        self.elastic_deform = ElasticDeform() if deform else None

    def call(self, inputs):
        img = inputs[0]
        mask = inputs[1]

        # img_ = img / 255.0
        img_ = self.img_resize(img)
        img_ = self.img_flip(img_)

        mask_ = self.mask_resize(mask)
        mask_ = self.mask_flip(mask_)
        mask_ = np.where(mask_ > 0, 1, 0).astype(np.uint8)

        if self.elastic_deform is not None:
            img_, mask_ = self.elastic_deform(img_, mask_)

        pad = (self.unet_info.input_size[0] - img_.shape[0]) // 2
        img_ = np.pad(img_, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')

        return img_, self.mask_crop(mask_)


class ElasticDeform(keras.layers.Layer):
    def __init__(self, p: float=0.5, seed: int=42):
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


def get_image_file_names(folder: os.PathLike, ext: str = "tif") -> List[str]:
    return sorted([img_name for img_name in os.listdir(folder) if img_name.endswith(ext)])


class Sequence:
    def __init__(self, root_folder: os.PathLike, seq_id: int):
        self.root_folder = root_folder
        self.folder = '0' + str(seq_id)
        self.seq_id = seq_id
        self.train_image_names = get_image_file_names(os.path.join(self.root_folder, '0' + str(self.seq_id)))
        self.gold_truths = self.get_truth_image_file_names(GT)
        self.silver_truths = self.get_truth_image_file_names(ST)
        self.image_shape = None
        self.image_dtype = None
        self.ann_dtype = None
        self.__find_image_info()

    def __find_image_info(self):
        img_f, ann_f = self.gold_truths[0]
        img = io.imread(os.path.join(self.root_folder, self.folder, img_f))
        ann = io.imread(os.path.join(self.root_folder, self.folder + '_' + GT, ann_f))

        self.image_shape = img.shape
        self.image_dtype = img.dtype
        self.ann_dtype = ann.dtype

    def get_truth_image_file_names(self, truth: str = "GT") -> List[Tuple[str, str]]:
        folder = os.path.join('0' + str(self.seq_id) + '_' + truth, SEG)

        truths: List[Tuple[str, str]] = []

        for truth_file in os.listdir(os.path.join(self.root_folder, folder)):
            match = man_seg_re.match(truth_file)
            if match:
                number_str = match.group(1)
                truths.append(('t' + number_str + '.tif', os.path.join(SEG, truth_file)))
        return truths

    def __str__(self):
        lines = ["Sequence " + str(self.seq_id) + ":"]
        lines.append(f"{len(self.train_image_names)} images")
        lines.append(f"{len(self.gold_truths)} Gold Truth SEG annotation files")
        lines.append(f"{len(self.silver_truths)} Silver Truth SEG annotation files")

        return "\n".join(lines)

    def get_random_example(self, truth: str = "GT", deform: bool = False, max_deform: int = 50,
                               smooth: float = 2.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        truths = self.gold_truths if truth == GT else self.silver_truths
        sample = random.sample(truths, k=1)[0]
        img_fname = os.path.join(self.root_folder, self.folder, sample[0])
        ann_fname = os.path.join(self.root_folder, self.folder + '_' + truth, sample[1])
        print(
            f'Loading {os.path.basename(img_fname)} and its corresponding {truth} annotation {os.path.basename(ann_fname)}.')

        image = io.imread(img_fname)
        annotation = io.imread(ann_fname)

        if deform:
            shape = image.shape[:-2] if image.ndim > 2 else image.shape
            dx, dy = generate_displacements(shape, max_displace=max_deform, sigma=smooth)
            # print(f'{image.shape=}, {image.shape[:-2]=}')
            # print(f'{dx.shape=}, {dy.shape=}')
            # print(f'{(np.sum(np.abs(dx - dy)))=}')
            image = deform_image(image, dx, dy)
            annotation = deform_image(annotation, dx, dy, order=0)

        labeled = color.label2rgb(annotation, image, bg_label=0)
        return image, annotation, labeled
    
    def get_random_mask(self, truth: str=GT):
        truths = self.gold_truths if truth==GT else self.silver_truths
        example = random.choice(truths)
        path = example[1]
        return io.imread(path, as_gray=True)
    
    def get_random_image(self):
        path = os.path.join(self.root_folder, self.folder, random.choice(self.train_image_names))
        return io.imread(path, as_gray=True)
    
    def load(self, truth: str = GT) -> Tuple[np.ndarray, np.ndarray]:
        if truth not in [GT, ST]:
            return None, None

        truths = self.gold_truths if truth == GT else self.silver_truths

        img_folder = os.path.join(self.root_folder, self.folder)
        truth_folder = os.path.join(self.root_folder, self.folder + '_' + truth, SEG)

        X_shape = (len(truths),) + self.image_shape
        ann_shape = (len(truths),) + self.image_shape[:2]

        X = np.zeros(X_shape, dtype=self.image_dtype)
        y = np.zeros(ann_shape, dtype=self.ann_dtype)

        for i, (img_name, truth_name) in enumerate(truths):
            img = io.imread(os.path.join(img_folder, img_name))
            ann = io.imread(os.path.join(truth_folder, truth_name))

            X[i] = img
            y[i] = ann

        return X, y


class Dataset:
    def __init__(self, folder: os.PathLike):
        self.folder = folder
        sequence_nums = sorted(
            list({int(sequence_folder[1]) for sequence_folder in os.listdir(folder) if sequence_folder[0] == '0'}))
        self.sequences = [Sequence(folder, seq_id) for seq_id in sequence_nums]

    def __str__(self):
        lines = [f"Dataset {os.path.basename(self.folder)}, filepath {self.folder}"]
        lines.append(f"{len(self.sequences)} sequences")
        lines.append('-------------------------------------------------------------------------------------')
        for seq in self.sequences:
            lines.append(str(seq))
            lines.append('-------------------------------------------------------------------------------------')

        return "\n".join(lines)

    def __check_seq_id(self, seq_id: int):
        assert 0 < seq_id <= len(
            self.sequences), "Sequence ID should be greater than 0 and less than or equal to " + str(
            len(self.sequences))

    def show_random_annotation(self, seq_id: int, truth: str = GT, deform: bool = False, max_deform: int = 50,
                               smooth: float = 2.5):
        self.__check_seq_id(seq_id)

        self.sequences[seq_id - 1].show_random_annotation(truth, deform=deform, max_deform=max_deform, smooth=smooth)

    def load(self, seq_id: int, truth: str = GT) -> Tuple[np.ndarray, np.ndarray]:
        if truth not in [GT, ST]:
            return None, None

        seq = self.sequences[seq_id]

        return seq.load(truth)

    def get_data(self, unet_info: UNetDataInfo, seed: int = 42, truth: str='ST'):
        image_names = []
        ann_names = []

        assert truth in [GT, ST]

        for seq in self.sequences:
            truths = seq.silver_truths if truth == ST else seq.gold_truths
            seq_folder = os.path.join(seq.root_folder, seq.folder)
            ann_folder = os.path.join(seq.root_folder, seq.folder + '_' + truth)
            image_names.extend([os.path.join(seq_folder, ex[0]) for ex in truths])
            ann_names.extend([os.path.join(ann_folder, ex[1]) for ex in truths])

        return CTCSequence(image_names, ann_names, unet_info, deform=truth==ST)


def load_example(example: tf.Tensor):
    img_f = example[0]
    ann_f = example[1]

    img = tf.io.read_file(img_f)
    img = tfio.experimental.image.decode_tiff(img)


def print_dataset_info(folder: Union[os.PathLike, str]):
    if isinstance(folder, str):
        folder = PurePath(folder)


def generate_displacements(shape: Tuple[int, int], max_displace: int = 10, sigma: float = 2.5, n_points: int = 16) -> \
        Tuple[np.ndarray, np.ndarray]:
    r = max_displace
    dx, dy = tf.random.uniform(shape=(n_points, n_points), minval=-r,
                                maxval=r), tf.random.uniform(shape=(n_points, n_points), minval=-r, maxval=r)
    dx, dy = resize(dx, shape, order=1), resize(dy, shape, order=1)

    bx, by = nd.gaussian_filter(dx, sigma=sigma), nd.gaussian_filter(dy, sigma=sigma)
    return bx, by


def deform_image(img: np.ndarray, dx: np.ndarray, dy: np.ndarray, order: int = 1):
    shape = img.shape[:-2]
    x, y = np.mgrid[0:img.shape[1], 0:img.shape[0]]
    cx, cy = x + dx, y + dy

    warped = np.zeros_like(img)
    if warped.ndim == 3 and warped.shape[-1] > 1:
        warped[:, :, 0] = nd.map_coordinates(img[:, :, 0], [cx, cy], order=order, mode='mirror')
        warped[:, :, 1] = nd.map_coordinates(img[:, :, 1], [cx, cy], order=order, mode='mirror')
        warped[:, :, 2] = nd.map_coordinates(img[:, :, 2], [cx, cy], order=order, mode='mirror')
    else:
        warped = nd.map_coordinates(img[:, :, 0], [cx, cy], order=order, mode='mirror')
        warped = np.expand_dims(warped, axis=-1)
    return warped


def compute_last_img_size(sz, d) -> int:
    cs = sz
    for i in range(d-1):
        cs -= 4
        cs = cs // 2
    return cs - 4


def compute_correct_size(sz, d) -> int:
    pad = 2 * 4 * (d-1) + 2 * (d-1) + 4
    cs = sz + pad
    while compute_last_img_size(cs, d) % 2:
        cs += 2
    return cs


def get_crop_sizes(sz, d) -> List[int]:
    sizes: List[int] = []

    curr_img_size = compute_last_img_size(sz, d)
    for i in range(d-1):
        curr_img_size *= 2
        sizes.append(curr_img_size)
        curr_img_size -= 4
    return sizes


def get_output_map_size(sz, d) -> int:
    return get_crop_sizes(sz, d)[-1] - 4


def train_unet(dataset_path: str, image_size: Tuple[int, int], epochs: int=30, unet_depth: int=4, unet_filters: int=32) -> Tuple[unet.UNet, keras.callbacks.History]:
    dataset = Dataset(dataset_path)
    info = UNetDataInfo(image_size, unet_depth)

    train_data = dataset.get_data(info, truth=ST)
    val_data = dataset.get_data(info, truth=GT)

    model = unet.UNet(unet_depth, unet_filters, info.input_size)
    
    callbacks = [
        keras.callbacks.ModelCheckpoint('./unet_model.hdf', save_best_only=True),
    ]
    opt = keras.optimizers.SGD()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-2), loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    hist = model.fit(train_data, epochs=epochs, validation_data=val_data, callbacks=callbacks)

    return model, hist


CellSubimage = namedtuple('CellSubimage', ['label', 'image', 'mask'])


def extract_cell_subimages(image: np.ndarray, labels: np.ndarray) -> List[CellSubimage]:
    props: List[measure._regionprops.RegionProperties] = measure.regionprops(labels, image)

    subimages: List[CellSubimage] = []
    # KDIR = '/home/radoslav/PhD_prep/debug/'
    for region in props:
        lab_sub = np.where(region.image.copy(), 255, 0).astype(np.uint8)
        # io.imsave(os.path.join(KDIR, f'big_{region.label}.png'), (255 * (labels > 0)).astype(np.uint8))
        # io.imsave(os.path.join(KDIR, f'sub_{region.label}.png'), lab_sub)
        img_sub = region.intensity_image.copy()
        subimages.append(CellSubimage(region.label, img_sub, lab_sub))
    return subimages


def save_cell_subimages(subimages: List[CellSubimage], ident: int, folder: str, save: str='both'):
    save_image = save == 'both' or save == 'image'
    save_mask = save == 'both' or save == 'mask'

    base_name = 'N' + str(ident) + '_'
    for subimage in subimages:
        specific_name = base_name + str(subimage.label) + '_'
        image_name = specific_name + 'i.tiff'
        mask_name = specific_name + 'm.png'
        if save_image:
            io.imsave(os.path.join(folder, image_name), subimage.image)
        if save_mask:
            io.imsave(os.path.join(folder, mask_name), subimage.mask)