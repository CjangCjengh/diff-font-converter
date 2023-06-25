import argparse
import os

import numpy as np
import torch
import torch.distributed as dist

from utils import dist_util, logger
from utils.script_util import (
    model_and_diffusion_defaults,
    args_to_dict,
    create_model_and_diffusion,
)
from utils.image_datasets import _list_image_files_recursively
from PIL import Image
import json


def img_pre_pros(img_path, image_size):
    pil_image = Image.open(img_path).resize((image_size, image_size))
    pil_image.load()
    pil_image = pil_image.convert("RGB")
    arr = np.array(pil_image)
    arr = arr.astype(np.float32) / 127.5 - 1
    return np.transpose(arr, [2, 0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default='./cfg/test_cfg.json',
                        help='config file path')
    parser = parser.parse_args()
    with open(parser.cfg_path, 'r') as f:
        cfg = json.load(f)
    cfg = create_cfg(cfg)
    model_path = cfg['model_path']
    img_save_path = cfg['img_save_path']
    source_dir = cfg['source_dir']
    batch_size = cfg['batch_size']
    num_fonts = cfg['num_fonts']
    if num_fonts > 1:
        nth_font = cfg['nth_font']

    dist_util.setup_dist()

    if not os.path.exists(img_save_path):
        os.mkdir(img_save_path)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(cfg, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if cfg['use_fp16']:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    noise = None

    # get source images
    src_img_paths = _list_image_files_recursively(source_dir)

    ch_idx = 0
    while ch_idx < len(src_img_paths):
        model_kwargs = {}
        img_paths = src_img_paths[ch_idx:ch_idx+batch_size]
        model_kwargs["y"] = [torch.tensor(img_pre_pros(img_path, cfg['image_size'])) for img_path in img_paths]
        model_kwargs["y"] = torch.stack(model_kwargs["y"]).to(dist_util.dev())
        if num_fonts > 1:
            model_kwargs['style'] = torch.tensor([nth_font] * len(model_kwargs['y'])).to(dist_util.dev())

        sample_fn = (
            diffusion.p_sample_loop if not cfg['use_ddim'] else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (len(model_kwargs["y"]), 3, cfg['image_size'], cfg['image_size']),
            clip_denoised=cfg['clip_denoised'],
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            noise=noise,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        for idx, img_sample in enumerate(sample):
            img = Image.fromarray(img_sample.cpu().numpy()).convert("RGB")
            img_name = "%s.png" % (os.path.basename(img_paths[idx]).split('.')[0])
            img.save(os.path.join(img_save_path, img_name))

        logger.log(f"created {ch_idx + len(sample)} samples")
        ch_idx += batch_size

    dist.barrier()
    logger.log("sampling complete")


def create_cfg(cfg):
    defaults = dict(
        clip_denoised=True,
        num_samples=100,
        batch_size=16,
        use_ddim=False,
        model_path="",
        cont_scale=1.0,
        sk_scale=1.0,
        sty_img_path="",
        stroke_path=None,
        attention_resolutions='40, 20, 10',
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(cfg)
    return defaults


if __name__ == "__main__":
    main()
