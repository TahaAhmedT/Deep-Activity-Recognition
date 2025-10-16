"""
This module defines the B3Dataset class for loading and preprocessing volleyball video frames and their annotations.
It collects cropped player images from annotated bounding boxes for use in deep learning activity recognition baselines.
"""

from src.utils.logging_utils import setup_logger
from src.utils.config_utils import load_config
from src.Preprocessing.volleyball_annot_loader import load_tracking_annot, load_video_annot

from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

CONFIG = load_config()

class B3Dataset(Dataset):
    """Custom Dataset for loading volleyball video frames and their bounding box annotations.

    This dataset loads images, crops player regions based on annotation files, and returns image-label pairs.
    """

    def __init__(self, videos_root: str, target_videos: list[int], annot_root, transform=None, verbose=False):
        """Initializes the B3Dataset.

        Args:
            videos_root (str): Root directory containing video folders.
            target_videos (list[int]): List of video indices to include.
            annot_root (str): Root directory containing annotation files.
            transform (callable, optional): Optional transform to be applied on a sample.
            verbose (bool, optional): If True, logs will be printed to console.
        """
        self.videos_root = videos_root
        self.target_videos = target_videos
        self.annot_root = annot_root
        self.verbose = verbose
        self.transform = transform
        self.dataset = []
        self.logger = setup_logger(
            log_file=__file__,
            log_dir="logs/baselines_logs/baseline3_logs",
            log_to_console=self.verbose,
            use_tqdm=True
        )
        self.logger.info("Initializing B3Dataset Module...")
        self.get_images_paths_labels()
        self.logger.info(f"B3Dataset Module Initialized with {len(self.dataset)} Samples!")
    

    def get_images_paths_labels(self):
        """Collects image paths and corresponding labels from the dataset.

        Iterates through the specified videos and clips, loads annotation files,
        crops player images using bounding boxes, and stores them with their labels.
        """
        self.logger.info("Collecting Images' paths and labels...")

        videos_dirs = os.listdir(self.videos_root)
        videos_dirs.sort()

        # Iterate on each video and for each video iterate on each clip
        for idx, video_dir in enumerate(videos_dirs):
            if idx in self.target_videos:
                video_dir_path = os.path.join(self.videos_root, video_dir)

                if not os.path.isdir(video_dir_path):
                    continue

                clips_dir = os.listdir(video_dir_path)
                clips_dir.sort()

                for clip_dir in clips_dir:
                    clip_dir_path = os.path.join(video_dir_path, clip_dir)

                    if not os.path.isdir(clip_dir_path):
                        continue

                    annot_file = os.path.join(self.annot_root, video_dir, clip_dir, f"{clip_dir}.txt")

                    frame_boxes = load_tracking_annot(annot_file)

                    for frame_id, boxes_info in frame_boxes.items():
                        try:
                            img_path = os.path.join(clip_dir_path, f"{frame_id}.jpg")
                            image = Image.open(img_path).convert("RGB")

                            for box_info in boxes_info:
                                x1, y1, x2, y2 = box_info.box
                                cropped_image = image.crop((x1, y1, x2, y2))
                                label = CONFIG["ACTIONS_DICT"][box_info.category]
                                self.dataset.append((cropped_image, label))

                        except Exception as e:
                            print(f"An error occurred: {e}")


    def __len__(self):
        """Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.dataset)
    

    def __getitem__(self, index):
        """Retrieves the image and label at the specified index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, label) where image is a PIL Image or transformed image, and label is the category.

        Raises:
            IndexError: If the index is out of range.
        """
        if index < 0 or index >= len(self.dataset):
            raise IndexError("Index out of range")
        img, label = self.dataset[index]

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

    def __init__(self, output_file, videos_root, target_videos):
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
        self.dataset = []
        self.logger = setup_logger(
            log_file=__file__,
            log_dir=os.path.join(CONFIG["LOGS_PATH"], "baseline3_logs"),
            log_to_console=CONFIG['verbose'],
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

        # Iterate on each video, in each video, iterate on each clip
        for idx, video_dir in enumerate(videos_dirs):
            if idx in self.target_videos:
                video_dir_path = os.path.join(self.output_file, video_dir)
                video_annot = os.path.join(self.videos_root, video_dir, 'annotations.txt')
                clip_category_dict = load_video_annot(video_annot)

                if not os.path.isdir(video_dir_path):
                    continue

                clips_dir = os.listdir(video_dir_path)
                clips_dir.sort()

                for clip_dir in clips_dir:
                    clip_features_file = os.path.join(video_dir_path, f"{clip_dir}.npy")

                    clip_label = CONFIG["CATEGORIES_DICT"][clip_category_dict[clip_dir]]
                    clip_features = np.load(clip_features_file) # Load all frames
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
    dataset = B3Dataset(
        videos_root=CONFIG["DATA_PATHS"]["videos_root"],
        target_videos=[1],
        annot_root=CONFIG["DATA_PATHS"]["annot_root"]
    )
    print(len(dataset))