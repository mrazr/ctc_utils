import os
import pathlib
import random
import re
import typing
from typing import List, Tuple

import numpy as np
from skimage import io, color
import tensorflow as tf

from img_utils import *


GT = "GT"
ST = "ST"
SEG = "SEG"
TRA = "TRA"
RES = "RES"

man_track_re = re.compile('man_track(\d{3,4}).tif')
man_seg_re = re.compile('man_seg(\d{3,4}).tif')


def get_image_file_names(folder: os.PathLike, ext: str = "tif") -> List[str]:
    return sorted([img_name for img_name in os.listdir(folder) if img_name.endswith(ext)])


class Sequence:
    def __init__(self, root_folder: os.PathLike, seq_id: int):
        self.root_folder = root_folder
        self.folder = '0' + str(seq_id)
        self.seq_id = seq_id
        self.train_image_names = get_image_file_names(os.path.join(self.root_folder, '0' + str(self.seq_id)))

        self.gold_truth_folder = os.path.join(self.root_folder, '0' + str(self.seq_id) + '_' + GT)
        self.gold_truths = self.get_truth_image_file_names(GT)

        self.silver_truth_folder = os.path.join(self.root_folder, '0' + str(self.seq_id) + '_' + ST)
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
        folder = self.gold_truth_folder if truth == GT else self.silver_truth_folder

        truths: List[Tuple[str, str]] = []

        for truth_file in os.listdir(os.path.join(folder, SEG)):
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

    def get_random_mask(self, truth: str = GT):
        truths = self.gold_truths if truth == GT else self.silver_truths
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

    def get_abs_truth_file_names(self, truth: str=ST) -> List[Tuple[str, str]]:
        folder = self.gold_truth_folder if truth == GT else self.silver_truth_folder
        truths = self.gold_truths if truth == GT else self.silver_truths
        imgs_anns: List[Tuple[str, str]] = [(os.path.join(self.root_folder, self.folder, img_name), os.path.join(folder, ann)) for img_name, ann in truths]
        return imgs_anns


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

    def train_val_split(self, train_split: float=0.3):
        image_paths: typing.List[pathlib.Path] = []
        ann_paths: typing.List[pathlib.Path] = []

        for seq in self.sequences:
            examples: typing.List[typing.Tuple[str, str]] = seq.silver_truths
            image_paths.extend([pathlib.Path(self.folder) / seq.folder / ex[0] for ex in examples])
            ann_paths.extend([pathlib.Path(self.folder) / f'{seq.folder}_{ST}' / ex[1] for ex in examples])

        indexes = list(range(len(image_paths)))
        random.shuffle(indexes)

        image_paths = [image_paths[idx] for idx in indexes]
        ann_paths = [ann_paths[idx] for idx in indexes]

        split_idx = int(train_split * len(image_paths))

        X_train = image_paths[split_idx:]
        y_train = ann_paths[split_idx:]

        X_val = image_paths[:split_idx]
        y_val = ann_paths[:split_idx]

        return X_train, y_train, X_val, y_val

    def test_data(self):
        image_paths: typing.List[pathlib.Path] = []
        ann_paths: typing.List[pathlib.Path] = []

        for seq in self.sequences:
            examples: typing.List[typing.Tuple[str, str]] = seq.gold_truths
            image_paths.extend([pathlib.Path(self.folder) / seq.folder / ex[0] for ex in examples])
            ann_paths.extend([pathlib.Path(self.folder) / f'{seq.folder}_ {GT}' / ex[1] for ex in examples])

        indexes = list(range(len(image_paths)))
        random.shuffle(indexes)

        image_paths = [image_paths[idx] for idx in indexes]
        ann_paths = [ann_paths[idx] for idx in indexes]

        return image_paths, ann_paths

    def image_paths(self) -> List[pathlib.Path]:
        image_paths: List[pathlib.Path] = []

        for seq in self.sequences:
            seq_path = pathlib.Path(seq.root_folder) / seq.folder
            image_paths.extend([seq_path / img_name for img_name in seq.train_image_names])

        return image_paths


class ImageImageSequence(tf.keras.utils.Sequence):
    def __init__(self, paths: List[pathlib.Path], image_size: typing.Tuple[int, int], shuffle: bool = False):
        super().__init__()
        self.image_size = image_size
        self.shuffle = shuffle
        self.image_paths: List[pathlib.Path] = paths
        if self.shuffle:
            random.shuffle(self.image_paths)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, item: int) -> typing.Tuple[np.ndarray, np.ndarray]:
        path = self.image_paths[item]
        img = io.imread(str(path))
        img = resize(img, self.image_size, order=2, preserve_range=True)
        if len(img.shape) == 3:
            img = img[np.newaxis, :, :]
        else:
            img = img[np.newaxis, :, :, np.newaxis]

        return img, img

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.image_paths)
