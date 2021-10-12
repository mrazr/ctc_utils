import os
from pathlib import PurePath
from typing import Union, List, Tuple
import re
from skimage import io, color
import random
import matplotlib.pyplot as plt


GT = "GT"
ST = "ST"
SEG = "SEG"
TRA = "TRA"
RES = "RES"

man_track_re = re.compile('man_track(\d{3,4}).tif')
man_seg_re = re.compile('man_seg(\d{3,4}).tif')


def get_image_file_names(folder: os.PathLike, ext: str="tif") -> List[str]:
  return sorted([img_name for img_name in os.listdir(folder) if img_name.endswith(ext)])


class CTCSequence:
  def __init__(self, root_folder: os.PathLike, seq_id: int):
    self.root_folder = root_folder
    self.folder = '0' + str(seq_id)
    self.seq_id = seq_id
    self.train_image_names = get_image_file_names(os.path.join(self.root_folder, '0' + str(self.seq_id)))
    self.gold_truths = self.get_truth_image_file_names(GT)
    self.silver_truths = self.get_truth_image_file_names(ST)

  def get_truth_image_file_names(self, truth: str="GT") -> List[Tuple[str, str]]:
    folder = os.path.join('0' + str(self.seq_id) + '_' + truth, SEG)

    truths: List[Tuple[str, str]] = []

    for truth_file in os.listdir(os.path.join(self.root_folder, folder)):
      match = man_seg_re.match(truth_file)
      if match:
        number_str = match.group(1)
        truths.append(('t' + number_str + '.tif', truth_file))
    return truths
  
  def __str__(self):
    lines = ["Sequence " + str(self.seq_id) + ":"]
    lines.append(f"{len(self.train_image_names)} images")
    lines.append(f"{len(self.gold_truths)} Gold Truth SEG annotation files")
    lines.append(f"{len(self.silver_truths)} Silver Truth SEG annotation files")

    return "\n".join(lines)
  
  def show_random_annotation(self, truth: str="GT"):
    truths = self.gold_truths if truth == GT else self.silver_truths
    sample = random.sample(truths, k=1)[0]
    img_fname = os.path.join(self.root_folder, self.folder, sample[0])
    ann_fname = os.path.join(self.root_folder, self.folder + '_' + truth, SEG, sample[1])
    print(f'Loading {os.path.basename(img_fname)} and its corresponding {truth} annotation {os.path.basename(ann_fname)}.')

    image = io.imread(img_fname)
    annotation = io.imread(ann_fname)


    labeled = color.label2rgb(annotation, image, bg_label=0)
    plt.imshow(labeled)
    plt.show()
    # io.imshow(labeled)


class CTCDataset:
  def __init__(self, folder: os.PathLike):
    self.folder = folder
    sequence_nums = sorted(list({int(sequence_folder[1]) for sequence_folder in os.listdir(folder) if sequence_folder[0] == '0'}))
    self.sequences = [CTCSequence(folder, seq_id) for seq_id in sequence_nums]
  
  def __str__(self):
    lines = [f"Dataset {self.folder}"]
    lines.append(f"{len(self.sequences)} sequences")
    lines.append('-------------------------------------------------------------------------------------')
    for seq in self.sequences:
      lines.append(str(seq))
      lines.append('-------------------------------------------------------------------------------------')
    
    return "\n".join(lines)

  def show_random_annotation(self, seq_id: int, truth: str=GT):
    assert 0 < seq_id <= len(self.sequences), "Sequence ID should be greater than 0 and less than or equal to " + str(len(self.sequences))

    self.sequences[seq_id - 1].show_random_annotation(truth)



def print_dataset_info(folder: Union[os.PathLike, str]):
  if isinstance(folder, str):
    folder = PurePath(folder)
  


