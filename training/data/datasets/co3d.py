import gzip
import json
import os
import os.path as osp
import logging
from typing import List, Optional

import cv2
import random
import numpy as np
import torch

from data.dataset_util import *
from data.base_dataset import BaseDataset

SEEN_CATEGORIES = [
    "apple", "backpack", "banana", "baseballbat", "baseballglove", "bicycle",
    "bottle", "bowl", "broccoli", "cake", "car", "carrot", "cellphone",
    "chair", "couch", "cup", "donut", "frisbee", "hairdryer", "handbag",
    "hotdog", "hydrant", "keyboard", "kite", "laptop", "microwave", "motorcycle",
    "mouse", "orange", "parkingmeter", "pizza", "plant", "remote", "sandwich",
    "skateboard", "stopsign", "suitcase", "teddybear", "toaster", "toilet",
    "toybus", "toyplane", "toytrain", "toytruck", "tv", "umbrella", "vase", "wineglass"
]

class Co3dDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        CO3D_DIR: str = None,
        CO3D_ANNOTATION_DIR: str = None,
        min_num_images: int = 4,
        len_train: int = 100000,
        len_test: int = 10000,
        selected_categories: Optional[List[str]] = None,
    ):
        super().__init__(common_conf=common_conf)
        self.debug = getattr(common_conf, "debug", False)
        self.training = getattr(common_conf, "training", True)
        self.load_depth = getattr(common_conf, "load_depth", False)
        self.inside_random = getattr(common_conf, "inside_random", False)
        self.allow_duplicate_img = getattr(common_conf, "allow_duplicate_img", True)

        if CO3D_DIR is None or CO3D_ANNOTATION_DIR is None:
            raise ValueError("Both CO3D_DIR and CO3D_ANNOTATION_DIR must be specified.")

        self.CO3D_DIR = CO3D_DIR
        self.CO3D_ANNOTATION_DIR = CO3D_ANNOTATION_DIR
        self.min_num_images = min_num_images

        # If debug, narrow down to 'cup' to speed up init
        categories = ["cup"] if self.debug else (sorted(selected_categories) if selected_categories else sorted(SEEN_CATEGORIES))
        split_name_list = ["train"] if split == "train" else ["test"]
        
        self.data_store = {}
        total_frame_num = 0

        for c in categories:
            for split_name in split_name_list:
                annotation_file = osp.join(CO3D_ANNOTATION_DIR, f"{c}_{split_name}.jgz")
                if not osp.exists(annotation_file):
                    continue

                try:
                    with gzip.open(annotation_file, "r") as fin:
                        annotation = json.loads(fin.read())
                except Exception as e:
                    logging.error(f"Error reading {annotation_file}: {e}")
                    continue

                for seq_name, seq_data in annotation.items():
                    # Filter frames that have the required camera parameters
                    valid_frames = [
                        frame for frame in seq_data 
                        if all(k in frame for k in ["R", "T", "focal_length", "principal_point"])
                    ]
                    
                    if len(valid_frames) < min_num_images:
                        continue
                    
                    total_frame_num += len(valid_frames)
                    self.data_store[seq_name] = valid_frames

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = total_frame_num

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: Co3D Data size: {self.sequence_list_len} valid sequences")
        logging.info(f"{status}: Co3D Data total valid frames: {self.total_frame_num}")

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        if self.sequence_list_len == 0:
            raise ValueError("No valid sequences found in annotation files.")

        # Retry loop to handle missing files on disk
        max_retries = 20
        for _ in range(max_retries):
            # Resolve sequence index
            if self.inside_random or seq_index is None:
                curr_idx = random.randint(0, self.sequence_list_len - 1)
            else:
                curr_idx = seq_index % self.sequence_list_len

            curr_seq_name = seq_name if seq_name else self.sequence_list[curr_idx]
            metadata = self.data_store[curr_seq_name]

            # Resolve image IDs
            curr_ids = ids if ids else np.random.choice(len(metadata), img_per_seq, replace=self.allow_duplicate_img)
            annos = [metadata[i] for i in curr_ids]
            
            target_image_shape = self.get_target_shape(aspect_ratio)

            images, depths, cam_points, world_points = [], [], [], []
            point_masks, extrinsics, intrinsics = [], [], []
            image_paths, original_sizes = [], []

            for anno in annos:
                img_rel_path = anno["filepath"]
                image_path = osp.join(self.CO3D_DIR, img_rel_path)
                
                if not osp.exists(image_path):
                    continue
                
                image = read_image_cv2(image_path)
                if image is None:
                    continue
                
                h, w = image.shape[:2]
                original_size = np.array([h, w])

                # Construct Extrinsic (4x4)
                R = np.array(anno["R"])
                T = np.array(anno["T"])
                extri_opencv = np.eye(4)
                extri_opencv[:3, :3] = R
                extri_opencv[:3, 3] = T

                # Construct Intrinsic (3x3)
                fl = anno["focal_length"]
                pp = anno["principal_point"]
                intri_opencv = np.array([
                    [fl[0], 0, pp[0]],
                    [0, fl[1], pp[1]],
                    [0, 0, 1]
                ])

                if self.load_depth:
                    depth_path = image_path.replace("/images", "/depths") + ".geometric.png"
                    depth_map = read_depth(depth_path, 1.0) if osp.exists(depth_path) else None
                else:
                    depth_map = None

                (p_img, p_depth, p_ext, p_int, w_pts, c_pts, mask, _) = self.process_one_image(
                    image, depth_map, extri_opencv, intri_opencv, original_size, target_image_shape, filepath=img_rel_path
                )

                images.append(p_img)
                depths.append(p_depth)
                extrinsics.append(p_ext)
                intrinsics.append(p_int)
                cam_points.append(c_pts)
                world_points.append(w_pts)
                point_masks.append(mask)
                image_paths.append(image_path)
                original_sizes.append(original_size)

            # If we found at least one image, we can return. 
            # Note: Model might expect exactly 'img_per_seq' frames; 
            # if so, we ensure len(images) == img_per_seq
            if len(images) > 0:
                return {
                    "seq_name": "co3d_" + curr_seq_name,
                    "ids": curr_ids,
                    "frame_num": len(extrinsics),
                    "images": images,
                    "depths": depths,
                    "extrinsics": extrinsics,
                    "intrinsics": intrinsics,
                    "cam_points": cam_points,
                    "world_points": world_points,
                    "point_masks": point_masks,
                    "original_sizes": original_sizes,
                    "image_paths": image_paths,
                }
            
            # Reset variables for next retry iteration
            seq_name = None
            ids = None

        raise RuntimeError(f"Data loading failed after {max_retries} attempts. Check CO3D_DIR paths.")
