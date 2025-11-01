"""
Helper datasets used across baselines and preprocessing for the volleyball activity recognition project.

This module provides:
- ImagesDataset: loads and returns image-level samples (either central frames per clip or cropped player images)
  along with their labels for training and evaluation of image-based models.
- FeaturesDataset: loads pre-extracted frame-level feature vectors saved as .npy files and pairs them with
  clip-level labels.

These dataset classes are intended to be used with torch.utils.data.DataLoader.
"""
from src.utils.logging_utils import setup_logger
from src.utils.config_utils import load_config
from src.Preprocessing.volleyball_annot_loader import load_tracking_annot, load_video_annot

from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np


class ImagesDataset(Dataset):
    """Dataset for loading images and their labels from the volleyball dataset.

    The dataset supports two modes:
      - image_level=True: sample representative central frames from each clip and return their file paths
      - image_level=False: load frames, crop player bounding boxes from tracking annotations and
        return cropped PIL.Image objects for every detected player instance

    The expected directory layout:
      <videos_root>/<video_dir>/<clip_id>/<frame_id>.jpg
    and (for bounding boxes):
      <annot_root>/<video_dir>/<clip_id>/<clip_id>.txt

    Attributes:
        videos_root (str): Root directory containing video folders.
        target_videos (list[int]): List of video indices to include.
        annot_root (str): Root directory containing per-clip tracking annotations.
        log_dir (str): Directory for logger output.
        image_level (bool): If True, operate at clip-level images; otherwise use per-player crops.
        actions_dict (dict|None): Mapping from original annotation category ids to model labels (used in crop mode).
        transform (callable|None): Optional transform applied to images on __getitem__.
        verbose (bool): If True, enable console logging.
        dataset (list): Internal list storing (image_or_path, label) tuples.
        logger (logging.Logger): Logger instance.
    """
    def __init__(self,
                 videos_root: str,
                 target_videos: list[int],
                 annot_root,
                 log_dir,
                 image_level: bool,
                 actions_dict: dict,
                 transform=None,
                 verbose=False):
        
        self.videos_root = videos_root
        self.target_videos = target_videos
        self.annot_root = annot_root
        self.image_level = image_level
        self.verbose = verbose
        self.transform = transform
        self.actions_dict = actions_dict
        self.log_dir = log_dir
        self.dataset = []
        self.logger = setup_logger(
            log_file=__file__,
            log_dir=self.log_dir,
            log_to_console=self.verbose,
            use_tqdm=True
        )
        self.logger.info("Initializing The Dataset Module...")
        self.get_images_paths_labels()
        self.logger.info(f"The Dataset Module Initialized with {len(self.dataset)} Samples!")
    

    def get_images_paths_labels(self):
        """Collect and index image paths and labels for the dataset.

        Scans the videos_root for videos listed in self.target_videos and, for each
        clip, either:
          - image_level=True: selects a small window around the central frames of the clip
            and stores (image_path, label) pairs; the clip-level label is read from
            the clip category mapping returned by load_video_annot, and then mapped
            through CONFIG["CATEGORIES_DICT"].
          - image_level=False: loads the per-clip tracking annotation (bounding boxes)
            via load_tracking_annot, opens each referenced frame, crops each player
            bounding box, maps the annotation category using self.actions_dict, and
            stores (PIL.Image, label) pairs.

        The discovered pairs are appended to self.dataset.
        """
        self.logger.info("Collecting Images' paths and labels...")

        videos_dirs = os.listdir(self.videos_root)
        videos_dirs.sort()

        # Iterate on each video and for each video iterate on each clip
        for idx, video_dir in enumerate(videos_dirs):
            if idx in self.target_videos:
                video_dir_path = os.path.join(self.videos_root, video_dir)

                if self.image_level:
                    video_annot = os.path.join(video_dir_path, 'annotations.txt')
                    clip_category_dict = load_video_annot(video_annot)

                if not os.path.isdir(video_dir_path):
                    continue

                clips_dir = os.listdir(video_dir_path)
                clips_dir.sort()

                for clip_dir in clips_dir:
                    clip_dir_path = os.path.join(video_dir_path, clip_dir)

                    if not os.path.isdir(clip_dir_path):
                        continue

                    if self.image_level:
                        clip_dir_path = os.path.join(video_dir_path, clip_dir)

                        if not os.path.isdir(clip_dir_path):
                            continue
                        
                        imgs_list = os.listdir(clip_dir_path)
                        mid_idx = len(imgs_list) // 2

                        for img in imgs_list[mid_idx-4: mid_idx+5]:
                            img_path = os.path.join(clip_dir_path, img)
                            label = CONFIG["CATEGORIES_DICT"][clip_category_dict[clip_dir]]
                            self.dataset.append((img_path, label))

                        self.logger.info(f"Collected {len(self.dataset)} image-label pairs.")
                    
                    else:
                        clip_annot = os.path.join(self.annot_root, video_dir, clip_dir, f"{clip_dir}.txt")
                        frame_boxes = load_tracking_annot(clip_annot)

                        for frame_id, boxes_info in frame_boxes.items():
                            try:
                                img_path = os.path.join(clip_dir_path, f"{frame_id}.jpg")
                                image = Image.open(img_path).convert("RGB")

                                for box_info in boxes_info:
                                    x1, y1, x2, y2 = box_info.box
                                    cropped_image = image.crop((x1, y1, x2, y2))
                                    label = self.actions_dict[box_info.category]
                                    self.dataset.append((cropped_image, label))

                            except Exception as e:
                                print(f"An error occurred: {e}")


    def __len__(self):
        """Return the number of indexed samples.

        Returns:
            int: Number of (image_or_path, label) pairs stored in self.dataset.
        """
        return len(self.dataset)
    

    def __getitem__(self, index):
        """Retrieve a sample by index.

        Args:
            index (int): Index of the desired sample in the dataset.

        Returns:
            tuple: (image, label)
                - If image_level=True the returned `image` is the image file path (str).
                - If image_level=False the returned `image` is a PIL.Image (or transformed tensor
                  if self.transform is provided).
        """
        if index < 0 or index >= len(self.dataset):
            raise IndexError("Index out of range")
        img, label = self.dataset[index]

        if self.image_level:
            img = Image.open(img).convert("RGB")

        if self.transform:
            img = self.transform(img)
        return img, label


class FeaturesDataset(Dataset):
    """Dataset for frame-level feature vectors extracted from clips.

    This dataset loads pre-extracted features saved as .npy files organized as:
        <output_file>/<video_id>/<clip_id>.npy

    Each .npy file is expected to have shape (num_frames, feature_dim). The dataset
    pairs each frame-level feature vector with the clip-level label obtained from the
    original video annotation file and returns items as (feature_vector, label).

    Attributes:
        output_file (str): Root directory containing extracted feature files organized by video.
        videos_root (str): Root directory of original video data (used to find annotation files).
        target_videos (list[int]): List of video indices to include.
        dataset (list): Internal list of (feature, label) tuples.
        logger (logging.Logger): Logger instance for progress and debug messages.
    """

    def __init__(self, output_file, videos_root, target_videos, log_dir, verbose):
        """Initializes the FeaturesDataset.

        Args:
            output_file (str): Directory containing extracted features (organized by video).
            videos_root (str): Root directory containing original video folders and annotations.
            target_videos (list[int]): List of video indices to include in the dataset.

        The constructor builds an internal index of (feature, label) pairs by calling
        get_features_labels().
        """
        super().__init__()
        self.output_file = output_file
        self.videos_root = videos_root
        self.target_videos = target_videos
        self.log_dir = log_dir
        self.verbose = verbose
        self.dataset = []
        self.logger = setup_logger(
            log_file=__file__,
            log_dir=self.log_dir,
            log_to_console=self.verbose,
            use_tqdm=True
        )
        self.logger.info("Initializing Features Dataset...")
        self.get_features_labels()
        self.logger.info(f"Features Dataset Initialized with {len(self.dataset)} Samples!")

    
    def get_features_labels(self):
        """Populates the internal dataset with (feature, label) tuples.

        Scans the output_file directory for each video in target_videos, loads the clip-level
        .npy files, and associates each frame's feature vector with the corresponding clip label
        obtained from the video's annotation file.

        The expected directory structure:
            output_file/<video_id>/<clip_id>.npy

        Each .npy file is expected to have shape (num_frames, feature_dim).

        Raises:
            FileNotFoundError: If an expected .npy file or annotation file is missing.
        """
        self.logger.info("Collecting Images' features and labels...") 
        videos_dirs = os.listdir(self.output_file) 
        videos_dirs.sort() 

        for idx, video_dir in enumerate(videos_dirs): 
            if idx in self.target_videos: 
                video_dir_path = os.path.join(self.output_file, video_dir) 
                video_annot = os.path.join(self.videos_root, video_dir, 'annotations.txt') 
                clip_category_dict = load_video_annot(video_annot) 

                if not os.path.isdir(video_dir_path): 
                    continue 

                # Iterate directly over .npy files in the video directory
                clip_files = [f for f in os.listdir(video_dir_path) if f.endswith('.npy')]
                clip_files.sort()

                for clip_file in clip_files:
                    clip_id = os.path.splitext(clip_file)[0]  # remove ".npy"
                    clip_features_file = os.path.join(video_dir_path, clip_file)

                    # get the label for this clip
                    clip_label = CONFIG["CATEGORIES_DICT"][clip_category_dict[clip_id]]
                    clip_features = np.load(clip_features_file)

                    for frame_idx in range(clip_features.shape[0]): 
                        self.dataset.append((clip_features[frame_idx], clip_label))
    
    def __len__(self):
        """Returns the number of samples in the dataset.

        Returns:
            int: Total number of (feature, label) pairs indexed.
        """
        return len(self.dataset)
    
    def __getitem__(self, index):
        """Retrieves a (feature, label) pair by index.

        Args:
            index (int): Index of the desired sample.

        Returns:
            tuple: (feature_vector, label)

        Raises:
            IndexError: If index is out of range.
        """
        if index < 0 or index >= len(self.dataset):
            raise IndexError("Index out of range")
        img_repr, label = self.dataset[index]
        return img_repr, label

        
if __name__ == "__main__":
    # Example usage: Initialize the dataset and print its length.
    CONFIG = load_config()
    # dataset = ImagesDataset(videos_root=CONFIG["DATA_PATHS"]["videos_root"],
    #                         target_videos=[0],
    #                         annot_root=CONFIG["DATA_PATHS"]["annot_root"],
    #                         log_dir="logs/baselines_logs/baseline1_logs",
    #                         image_level=True,
    #                         actions_dict=CONFIG["CATEGORIES_DICT"]
    #                         )
    dataset = FeaturesDataset(
        output_file=CONFIG["DATA_PATHS"]["features_root"],
        videos_root=CONFIG["DATA_PATHS"]["videos_root"],
        target_videos=[0],
        log_dir="logs/baselines_logs/baseline1_logs",
        verbose=True
    )
    print(len(dataset))