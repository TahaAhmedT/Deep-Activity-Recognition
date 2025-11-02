"""
This script extracts deep features from volleyball player images using a fine-tuned ResNet-50 model.
It loads model checkpoints, applies preprocessing, and saves extracted features for each video clip.
"""

from src.utils.config_utils import load_config
from src.utils.checkpoints_utils import load_checkpoint
from src.Preprocessing.extract_features import extract_features
from src.Preprocessing.volleyball_annot_loader import load_video_annot
from src.utils.logging_utils import setup_logger
from src.helpers.finetune_helper import get_model

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights


def extract(log_dir: str, videos_root: str, train_ids: list[int], val_ids: list[int], annot_root: str, output_root: str, model, transform, image_level: bool, image_classify: bool, verbose: bool):
    """
    function to extract features from volleyball video clips.

    Iterates through videos and clips, loads annotations, and saves extracted features as .npy files.
    """
    logger = setup_logger(
            log_file=__file__,
            log_dir=log_dir,
            log_to_console=verbose,
            use_tqdm=True
        )

    videos_dirs = os.listdir(videos_root)
    videos_dirs.sort()

    # Iterate on each video and for each video iterate on each clip
    for idx, video_dir in enumerate(videos_dirs):
        if idx in train_ids or val_ids:
            logger.info(f"Working on Video Number: {idx}")
            video_dir_path = os.path.join(videos_root, video_dir)

            if not os.path.isdir(video_dir_path):
                continue

            clips_dir = os.listdir(video_dir_path)
            clips_dir.sort()

            for clip_dir in clips_dir:
                clip_dir_path = os.path.join(video_dir_path, clip_dir)

                if not os.path.isdir(clip_dir_path):
                    continue

                annot_file = os.path.join(annot_root, video_dir, clip_dir, f"{clip_dir}.txt")
                output_file = os.path.join(output_root, video_dir)

                if not os.path.isdir(output_file):
                    os.makedirs(output_file)
                
                output_file = os.path.join(output_file, f"{clip_dir}.npy")
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                extract_features(clip_dir_path, annot_file, output_file, model, transform, device, image_level, image_classify)
