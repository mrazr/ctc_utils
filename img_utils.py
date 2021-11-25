import os
from collections import namedtuple
from typing import Tuple, List

import numpy as np
import scipy.ndimage as nd
from skimage import measure, io
from skimage.transform import resize


def generate_displacements(shape: Tuple[int, int], max_displace: int = 10, sigma: float = 2.5, n_points: int = 16) -> \
        Tuple[np.ndarray, np.ndarray]:
    r = max_displace
    dx, dy = np.random.uniform(size=(n_points, n_points), low=-r,
                               high=r), np.random.uniform(size=(n_points, n_points), low=-r, high=r)
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



def map_to_gs(img):
    mmin, mmax = np.min(img), np.max(img)
    range = mmax - mmin

    return (255 * np.array((img - mmin) / (range + 0.00001))).astype(np.uint8)
