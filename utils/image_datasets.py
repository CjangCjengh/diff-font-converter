import os
import math
import random
import json

from PIL import Image
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
from . import logger


def load_data(
    *,
    source_dir,
    target_dir,
    batch_size,
    image_size,
    deterministic=False,
    random_crop=False,
    random_flip=False,
):
    if not source_dir or not target_dir:
        raise ValueError("unspecified data directory")
    
    source_files = []
    for d in source_dir:
        source_files.extend(_list_image_files_recursively(d))
    target_files = []
    for i, d in enumerate(target_dir):
        files = _list_image_files_recursively(d)
        logger.log(f"found {len(files)} files in {d}")
        for f in files:
            target_files.append((f, i))

    dataset = ImageDataset(
        image_size,
        source_files,
        target_files,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
        multi_font=len(target_dir) > 1,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in os.listdir(data_dir):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif os.path.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        source_paths,
        target_paths,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        multi_font=False,
    ):
        super().__init__()
        self.resolution = resolution
        source_dict = {}
        for f in source_paths:
            k = int(os.path.basename(f).split(".")[0],16)
            if k not in source_dict:
                source_dict[k] = [f]
            else:
                source_dict[k].append(f)
        image_pairs = []
        for f, i in target_paths:
            k = int(os.path.basename(f).split(".")[0],16)
            if k in source_dict:
                image_pairs.append((source_dict[k],f,i))
        self.image_pairs = image_pairs[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.multi_font = multi_font

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        source_path, target_path, style = self.image_pairs[idx]
        source_path = random.choice(source_path)
        with open(source_path, "rb") as f:
            pil_image1 = Image.open(f)
            pil_image1.load()
        with open(target_path, "rb") as f:
            pil_image2 = Image.open(f)
            pil_image2.load()
        pil_image1 = pil_image1.convert("RGB")
        pil_image2 = pil_image2.convert("RGB")

        if self.random_crop:
            arr1 = random_crop_arr(pil_image1, self.resolution)
            arr2 = random_crop_arr(pil_image2, self.resolution)
        else:
            arr1 = center_crop_arr(pil_image1, self.resolution)
            arr2 = center_crop_arr(pil_image2, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr1 = arr1[:, ::-1]
            arr2 = arr2[:, ::-1]

        arr1 = arr1.astype(np.float32) / 127.5 - 1
        arr2 = arr2.astype(np.float32) / 127.5 - 1

        out_dict = {}
        out_dict['y'] = np.transpose(arr1, [2, 0, 1])
        if self.multi_font:
            out_dict['style'] = style

        return np.transpose(arr2, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
