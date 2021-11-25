import argparse
import os
from collections import namedtuple
from typing import List, Union

import numpy as np
from skimage import measure, io

from ctc_dataset import Dataset


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


def save_cell_subimages(subimages: List[CellSubimage], ident: Union[int, str], folder: str, save: str='both'):
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


def make_dir(fol):
    nonexistent = []
    current_path = os.path.normpath(fol)
    while not os.path.exists(current_path):
        nonexistent.append(current_path)
        current_path, _ = os.path.split(current_path)
    for path in reversed(nonexistent):
        os.mkdir(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract individual cell mask from segmentation masks for a given CTC folder.")
    parser.add_argument('ctc_folder', help='Cell tracking challenge dataset folder.')
    parser.add_argument('--save_pairs', action='store_true', help='if provided, extracts also the corresponding intensity subimage')
    parser.add_argument('--truth', default='GT', help='Kind of annotation to extract, either ST(silver) or GT(gold) truth', choices=['GT', 'ST'])
    parser.add_argument('--out', default='./', help='Folder where the extracted mask will be saved. Defaults to {current_dir}/{ctc_folder}_cell_masks/')

    args = parser.parse_args()

    in_path = os.path.abspath(args.ctc_folder)
    out_path = os.path.join(os.path.abspath(args.out), f'{os.path.split(in_path)[1]}_cell_masks_{args.truth}/')
    out_path = os.path.normpath(out_path)

    save_flag = 'both' if args.save_pairs else 'mask'
    truth = args.truth

    if not os.path.exists(out_path):
        make_dir(out_path)

    print(in_path)
    print(out_path)

    ds = Dataset(in_path)

    for seq in ds.sequences:
        seq_str = f'{seq.seq_id:0>2}'
        imgs_anns = seq.get_abs_truth_file_names(truth)
        for i, (img, ann) in enumerate(imgs_anns):
            img = io.imread(img, as_gray=True)
            ann = io.imread(ann, as_gray=True)

            subimages = extract_cell_subimages(img, ann)
            save_cell_subimages(subimages, seq_str + '_' + str(i), folder=out_path, save=save_flag)


