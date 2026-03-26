# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import cv2
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from .dataset_util import *
import logging

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class BaseDataset(Dataset):
    """
    Base dataset class for VGGT and VGGSfM training.

    Handles common operations like image resizing, augmentation,
    and coordinate transformations. Concrete datasets inherit
    from this class.
    """
    def __init__(self, common_conf, sequences=None, seq_dict=None, training=True):
        """
        Args:
            common_conf: Config object with shared dataset parameters
            sequences (list, optional): List of sequence info dicts
            seq_dict (dict, optional): Dict mapping seq_name -> seq_info
            training (bool): True for training, False for validation
        """
        super().__init__()
        self.img_size = common_conf.img_size
        self.patch_size = common_conf.patch_size
        self.aug_scale = getattr(common_conf.augs, "scales", [0.8, 1.2])
        self.rescale = common_conf.rescale
        self.rescale_aug = common_conf.rescale_aug
        self.landscape_check = common_conf.landscape_check
        self.common_conf = common_conf
        self.training = training

        # Dataset sequences
        self.sequences = sequences if sequences is not None else []
        self.seq_dict = seq_dict if seq_dict is not None else {}
        self.sequence_list = self.sequences
        self.len_train = len(self.sequence_list)

    def __len__(self):
        return self.len_train

    def __getitem__(self, idx):
        """
        Returns:
            dict: Sample as returned by get_data
        """
        # idx can be int or tuple/list of (seq_index, img_per_seq, aspect_ratio)
        if isinstance(idx, int):
            seq_index = idx
            img_per_seq = getattr(self.common_conf, "fix_img_num", 5)
            aspect_ratio = getattr(self.common_conf, "fix_aspect_ratio", 1.0)
        elif isinstance(idx, (list, tuple)) and len(idx) == 3:
            seq_index, img_per_seq, aspect_ratio = idx
        else:
            raise TypeError(
                f"idx must be int or tuple/list of 3 elements, got {type(idx)}"
            )

        return self.get_data(seq_index=seq_index, img_per_seq=img_per_seq, aspect_ratio=aspect_ratio)

    def get_data(self, seq_index=None, seq_name=None, ids=None, aspect_ratio=1.0, **kwargs):
        """
        Fetch a single training sample safely, skipping missing frames.
        """
        # Select sequence
        if seq_index is not None:
            seq_index = min(seq_index, len(self.sequence_list)-1)
            seq_info = self.sequence_list[seq_index]
        elif seq_name is not None and seq_name in self.seq_dict:
            seq_info = self.seq_dict[seq_name]
        else:
            seq_info = np.random.choice(self.sequences)

        # Sample image indices
        all_ids = ids if ids is not None else seq_info.get("image_ids", list(range(len(seq_info.get("image_paths", [])))))
        sampled_ids = []
        max_attempts = 10

        for img_idx in all_ids:
            attempts = 0
            while attempts < max_attempts:
                frame_valid = True
                # Check for missing extrinsics/intrinsics
                if "extrinsics" not in seq_info or "intrinsics" not in seq_info:
                    logging.warning(f"Skipping sequence {seq_info.get('seq_name','unknown')} missing extrinsics/intrinsics")
                    frame_valid = False
                    break
                if img_idx >= len(seq_info.get("image_paths", [])):
                    logging.warning(f"Skipping frame index {img_idx} out of range for sequence {seq_info.get('seq_name','unknown')}")
                    frame_valid = False
                    img_idx = np.random.choice(all_ids)
                if frame_valid:
                    sampled_ids.append(img_idx)
                    break
                attempts += 1

        images, depths, extrinsics, intrinsics, world_pts, cam_pts, masks, tracks = [], [], [], [], [], [], [], []

        for img_idx in sampled_ids:
            image_path = seq_info["image_paths"][img_idx]
            depth_paths = seq_info.get("depth_paths", [None]*len(seq_info["image_paths"]))
            depth_path = depth_paths[img_idx]

            extri_opencv = seq_info["extrinsics"][img_idx]
            intri_opencv = seq_info["intrinsics"][img_idx]

            image = cv2.imread(image_path)
            if image is None:
                logging.warning(f"Skipping missing/corrupt image: {image_path}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            depth_map = (
                cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                if depth_path is not None else np.zeros(image.shape[:2], dtype=np.float32)
            )

            original_size = np.array(image.shape[:2])

            # Process through VGGT pipeline
            img, depth_map, ext, intr, world, cam, mask, track = self.process_one_image(
                image=image,
                depth_map=depth_map,
                extri_opencv=extri_opencv,
                intri_opencv=intri_opencv,
                original_size=original_size,
                target_image_shape=self.get_target_shape(aspect_ratio)
            )

            images.append(img)
            depths.append(depth_map)
            extrinsics.append(ext)
            intrinsics.append(intr)
            world_pts.append(world)
            cam_pts.append(cam)
            masks.append(mask)
            tracks.append(track)

        return {
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "world_points": world_pts,
            "cam_points": cam_pts,
            "masks": masks,
            "tracks": tracks
        }

    def get_target_shape(self, aspect_ratio):
        """
        Target shape for vision transformer input
        """
        short_size = int(self.img_size * aspect_ratio)
        small_size = self.patch_size
        if short_size % small_size != 0:
            short_size = (short_size // small_size) * small_size
        return np.array([short_size, self.img_size])

    def process_one_image(
        self, image, depth_map, extri_opencv, intri_opencv,
        original_size, target_image_shape, track=None, filepath=None, safe_bound=4
    ):
        """
        Process image + depth + intrinsics/extrinsics
        """
        image = np.copy(image)
        depth_map = np.copy(depth_map)
        extri_opencv = np.copy(extri_opencv)
        intri_opencv = np.copy(intri_opencv)
        if track is not None:
            track = np.copy(track)

        # Augmentation
        if self.training and self.aug_scale:
            h_scale, w_scale = np.random.uniform(self.aug_scale[0], self.aug_scale[1], 2)
            h_scale = min(h_scale, 1.0)
            w_scale = min(w_scale, 1.0)
            aug_size = (original_size * np.array([h_scale, w_scale])).astype(np.int32)
        else:
            aug_size = original_size

        # Crop & adjust intrinsics
        image, depth_map, intri_opencv, track = crop_image_depth_and_intrinsic_by_pp(
            image, depth_map, intri_opencv, aug_size, track=track, filepath=filepath
        )

        # Handle landscape orientation
        rotate_to_portrait = False
        if self.landscape_check:
            if original_size[0] > 1.25 * original_size[1]:
                if (target_image_shape[0] != target_image_shape[1]) and (np.random.rand() > 0.5):
                    target_image_shape = np.array([target_image_shape[1], target_image_shape[0]])
                    rotate_to_portrait = True

        # Resize
        if self.rescale:
            image, depth_map, intri_opencv, track = resize_image_depth_and_intrinsic(
                image, depth_map, intri_opencv, target_image_shape, original_size,
                track=track, safe_bound=safe_bound, rescale_aug=self.rescale_aug
            )

        # Final crop
        image, depth_map, intri_opencv, track = crop_image_depth_and_intrinsic_by_pp(
            image, depth_map, intri_opencv, target_image_shape, track=track,
            filepath=filepath, strict=True
        )

        if rotate_to_portrait:
            clockwise = np.random.rand() > 0.5
            image, depth_map, extri_opencv, intri_opencv, track = rotate_90_degrees(
                image, depth_map, extri_opencv, intri_opencv, clockwise=clockwise, track=track
            )

        # Convert depth to world/camera points
        world_pts, cam_pts, mask = depth_to_world_coords_points(depth_map, extri_opencv, intri_opencv)

        return image, depth_map, extri_opencv, intri_opencv, world_pts, cam_pts, mask, track

    def get_nearby_ids(self, ids, full_seq_num, expand_ratio=None, expand_range=None):
        """
        Sample a set of nearby IDs within a sequence
        """
        if len(ids) == 0:
            raise ValueError("No IDs provided.")

        if expand_range is None and expand_ratio is None:
            expand_ratio = 2.0

        total_ids = len(ids)
        start_idx = ids[0]

        if expand_range is None:
            expand_range = int(total_ids * expand_ratio)

        low = max(0, start_idx - expand_range)
        high = min(full_seq_num, start_idx + expand_range)
        valid_range = np.arange(low, high)
        sampled_ids = np.random.choice(valid_range, size=(total_ids - 1), replace=True)
        result_ids = np.insert(sampled_ids, 0, start_idx)
        return result_ids
