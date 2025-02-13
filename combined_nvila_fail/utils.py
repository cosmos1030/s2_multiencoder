# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# dynamic_preprocess and find_closest_aspect_ratio are referenced from https://github.com/OpenGVLab/InternVL

import base64
import os
import tempfile
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from transformers import StoppingCriteria

from llava.constants import DEFAULT_IMAGE_TOKEN


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    """
    Expand the given PIL image to a square shape by adding padding.

    Parameters:
    - pil_img: The PIL image to be expanded.
    - background_color: The color of the padding to be added.

    Returns:
    - The expanded PIL image.

    If the image is already square, it is returned as is.
    If the image is wider than it is tall, padding is added to the top and bottom.
    If the image is taller than it is wide, padding is added to the left and right.
    """
    width, height = pil_img.size
    if pil_img.mode == "L":
        background_color = background_color[0]
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=384, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def dynamic_s2_preprocess(image, s2_scales=[384, 768, 1152], max_num=12, image_size=384):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    min_num = (s2_scales[-1] // s2_scales[0]) ** 2  # at least use number of tiles as the largest scale

    processed_images = []

    ##########################################################################################
    ############# Add tiles for all but the last scale using fixed squre ratio ###############
    ##########################################################################################

    for scale in s2_scales[:-1]:
        target_width = image_size * (scale // s2_scales[0])
        target_height = image_size * (scale // s2_scales[0])
        blocks = (scale // s2_scales[0]) ** 2

        # resize the image
        resized_img = image.resize((target_width, target_height))
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)

    ##########################################################################################
    ################ Add tiles for the last scale using dynamic aspect ratio #################
    ##########################################################################################

    # calculate the existing image aspect ratio
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    return processed_images, (target_aspect_ratio[1], target_aspect_ratio[0])


def dynamic_process_images_and_prompt(images, prompt, data_args, image_folder=None, max_tiles=None):
    prompt = prompt.split(DEFAULT_IMAGE_TOKEN)
    idx = 0
    all_images = []
    for img in images:
        processed_images = process_image(img, data_args, image_folder, enable_dynamic_res=True, max_tiles=max_tiles)
        all_images.append(processed_images)
        prompt.insert(idx + 1, f"{DEFAULT_IMAGE_TOKEN}\n" * processed_images.shape[0])
        idx += 2
    prompt = "".join(prompt)
    if all_images:
        all_images = torch.cat(all_images)
    else:
        all_images = None
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, "")
    return all_images, prompt


def dynamic_s2_process_images_and_prompt(images, prompt, data_args, image_folder=None):
    idx = 0
    all_images = []
    all_block_size = []
    for img in images:
        processed_images, block_size = process_image(img, data_args, image_folder, enable_dynamic_s2=True)
        all_images.append(processed_images)
        all_block_size.append(block_size)
        idx += 2
    if all_images:
        all_images = torch.cat(all_images)
    else:
        all_images = None
    return all_images, all_block_size


def process_image(
    image_file, data_args, image_folder, enable_dynamic_res=False, enable_dynamic_s2=False, max_tiles=None
):
    processor = data_args.image_processor
    if isinstance(image_file, str):
        if image_folder is not None:
            image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
    else:
        # image is stored in bytearray
        image = image_file
    image = image.convert("RGB")
    if hasattr(data_args.image_processor, "crop_size"):
        # CLIP vision tower
        crop_size = data_args.image_processor.crop_size
    else:
        # SIGLIP vision tower
        assert hasattr(data_args.image_processor, "size")
        crop_size = data_args.image_processor.size
    if "dynamic_s2" in data_args.image_aspect_ratio and enable_dynamic_s2:
        assert crop_size["height"] == crop_size["width"]
        images, block_size = dynamic_s2_preprocess(
            image, s2_scales=data_args.s2_scales, max_num=data_args.max_tiles, image_size=crop_size["height"]
        )
        images = [processor.preprocess(image, return_tensors="pt")["pixel_values"][0] for image in images]
        return torch.stack(images), block_size
    if "dynamic" in data_args.image_aspect_ratio and enable_dynamic_res:
        assert crop_size["height"] == crop_size["width"]
        if max_tiles is not None:
            max_num = max_tiles
        else:
            max_num = data_args.max_tiles
        images = dynamic_preprocess(image, min_num=data_args.min_tiles, max_num=max_num, image_size=crop_size["height"])
        images = [processor.preprocess(image, return_tensors="pt")["pixel_values"][0] for image in images]
        return torch.stack(images)

    if data_args.image_aspect_ratio == "resize":
        image = image.resize((crop_size["width"], crop_size["height"]))
    if data_args.image_aspect_ratio == "pad":

        def expand2square(pil_img, background_color):
            width, height = pil_img.size
            if width == height:
                return pil_img
            elif width > height:
                result = Image.new(pil_img.mode, (width, width), background_color)
                result.paste(pil_img, (0, (width - height) // 2))
                return result
            else:
                result = Image.new(pil_img.mode, (height, height), background_color)
                result.paste(pil_img, ((height - width) // 2, 0))
                return result

        image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
        image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    else:
        # Using default behavior of the vision encoder
        # For CLIP, default is central crop
        # For Radio, default is central crop
        # For Siglip, default is resize
        # For InternVIT, default is resize
        image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    return image


def process_images(images, image_processor, model_cfg, enable_dynamic_res=False):
    model_cfg.image_processor = image_processor
    new_images = [process_image(image, model_cfg, None, enable_dynamic_res=enable_dynamic_res) for image in images]

    if all(x.shape == new_images[0].shape for x in new_images):
        if len(new_images[0].shape) == 4:
            new_images = torch.cat(new_images, dim=0)
        elif len(new_images[0].shape) == 3:
            new_images = torch.stack(new_images, dim=0)
        else:
            raise ValueError(f"new_images rank does not equal to 4, rank: {len(new_images[0].shape)}")
    else:
        raise ValueError("The shape of images in new_images is different!")
    return new_images


def tokenizer_image_token(prompt, tokenizer, return_tensors=None):
    return tokenizer(prompt, return_tensors=return_tensors).input_ids[0]


def is_gemma_tokenizer(tokenizer):
    return "gemma" in tokenizer.__class__.__name__.lower()


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith("checkpoint-"):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0] :] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)

