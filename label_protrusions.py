import argparse
import os
from collections import namedtuple
from typing import List, Union
import shutil

import numpy as np
from skimage import measure, io
import skimage.morphology as M
import cv2 as cv

from ctc_dataset import Dataset


class Protrusion:
    def __init__(self, bbox, mask, coords, parent=None):
        self.bbox = bbox
        self.mask = mask
        self.parent = parent
        self.coords = coords
        self.children = []


def has_protrusions(mask):
    chull = M.convex_hull_image(mask)
    prots = np.logical_xor(chull, mask)
    prots = M.binary_opening(prots, M.disk(1))
    
    return np.count_nonzero(prots) > 3


def get_protrusions(protrusion: Union[np.ndarray, Protrusion]):
    # assert (cell_mask is not None or protrusion is not None), 'Must provide at least one of the two'
    # assert not (cell_mask is not None and protrusion is not None), 'Must provide only one of the two'
    
    if isinstance(protrusion, np.ndarray):
        mask = protrusion
    else:
        mask = protrusion.mask
        
    if not has_protrusions(mask):
        return None
    
    A = np.count_nonzero(mask)
    r = round(np.sqrt(A / np.pi) * 0.25)
    
    # show(mask)
    prots = cv.morphologyEx(mask.astype('uint8'), cv.MORPH_TOPHAT, M.disk(r))
    # show(prots)
    
    lab = measure.label(prots)
    props = measure.regionprops(lab)
    
    found_p = []
    
    for prop in props:
        if prop.area < 15:
            continue
        prot = Protrusion(prop.bbox, prop.image, prop.coords, protrusion if isinstance(protrusion, Protrusion) else None)
        found_p.append(prot)
        # show(prop.image)
    return found_p


def extract_protrusions(cell_mask: np.ndarray, levels: int=1):
    assert levels >= 1 and levels <= 3
    
    level = 0
    base_protrusions = get_protrusions(cell_mask)
    current_protrusions = base_protrusions
    
    for level in range(levels-1):
        new_protrusions = []
        for protrusion in current_protrusions:
            protrusion.children = get_protrusions(protrusion)
            if protrusion.children is not None:
                new_protrusions.extend(protrusion.children)
        current_protrusions = new_protrusions
    return base_protrusions


def copy_sequence_folders(dataset: Dataset, out_path):
    for seq in dataset.sequences:
        shutil.copytree(os.path.join(dataset.folder, seq.folder), os.path.join(out_path, seq.folder), dirs_exist_ok=True)


def make_dir(fol):
    nonexistent = []
    current_path = os.path.normpath(fol)
    while not os.path.exists(current_path):
        nonexistent.append(current_path)
        current_path, _ = os.path.split(current_path)
    for path in reversed(nonexistent):
        os.mkdir(path)


def relabel_protrusions(mask):
    protrusions = extract_protrusions(mask)
    if protrusions is None:
        return mask
    labels = mask.copy().astype('uint8')

    for prot in protrusions:
        coords = np.array(prot.coords)
        labels[coords[:,0], coords[:,1]] = 2
    return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Assign separate labels to cell protrusions.")
    parser.add_argument('ctc_folder', help='Cell tracking challenge dataset folder.')
    parser.add_argument('--out', default='./', help='Folder of the modified dataset.. Defaults to {current_dir}/{ctc_folder}_protrusions/')

    args = parser.parse_args()

    in_path = os.path.abspath(args.ctc_folder)
    out_path = os.path.join(os.path.abspath(args.out), f'{os.path.split(in_path)[1]}_protrusions/')
    out_path = os.path.normpath(out_path)

    if not os.path.exists(out_path):
        make_dir(out_path)
    
    dataset = Dataset(in_path)

    copy_sequence_folders(dataset, out_path)

    for seq in dataset.sequences:
        seg_f = os.path.join(dataset.folder, f'{seq.folder}_GT')

        seg_out = os.path.join(out_path, f'{seq.folder}_GT/SEG')
        if not os.path.exists(seg_out):
            make_dir(seg_out)
        for _, mask_fname in seq.gold_truths:
            mask = io.imread(os.path.join(seg_f, mask_fname), as_gray=True)
            mask_fname = os.path.split(mask_fname)[1]
            label = measure.label(mask)
            props = measure.regionprops(label)

            res_mask = mask.copy()
            for prop in props:
                cell_mask = prop.image
                relabeled = relabel_protrusions(cell_mask)
                coords = np.array(prop.coords)
                # res_mask[prop.bbox[0]:prop.bbox[2], prop.bbox[1]:prop.bbox[3]] = relabeled
                c2 = np.nonzero(relabeled)
                res_mask[coords[:, 0], coords[:, 1]] = relabeled[c2[0], c2[1]]
            io.imsave(os.path.join(seg_out, mask_fname), res_mask)

        seg_f = os.path.join(dataset.folder, f'{seq.folder}_ST')
        seg_out = os.path.join(out_path, f'{seq.folder}_ST/SEG')
        if not os.path.exists(seg_out):
            make_dir(seg_out)
        for _, mask_fname in seq.silver_truths:
            mask = io.imread(os.path.join(seg_f, mask_fname), as_gray=True)

            mask_fname = os.path.split(mask_fname)[1]
            label = measure.label(mask)
            props = measure.regionprops(label)

            res_mask = mask.copy()
            for prop in props:
                cell_mask = prop.image
                relabeled = relabel_protrusions(cell_mask)
                coords = np.array(prop.coords)
                # res_mask[prop.bbox[0]:prop.bbox[2], prop.bbox[1]:prop.bbox[3]] = relabeled
                c2 = np.nonzero(relabeled)
                res_mask[coords[:, 0], coords[:, 1]] = relabeled[c2[0], c2[1]]
            io.imsave(os.path.join(seg_out, mask_fname), res_mask)