from utils import setup_logger
from Preprocessing import load_video_annot

from torch.utils.data import Dataset
import os
import numpy as np


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

    def __init__(self, output_file, videos_root, target_videos, categories_dict, log_dir, crop: bool, sequence=False, verbose=False):
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
        self.categories_dict = categories_dict
        self.log_dir = log_dir
        self.crop = crop
        self.sequence = sequence
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
                    clip_label = self.categories_dict[clip_category_dict[clip_id]]
                    clip_features = np.load(clip_features_file)

                    if self.sequence and not self.crop:
                        seq_features = []
                        for frame_idx in range(clip_features.shape[0]):
                            seq_features.append(clip_features[frame_idx])
                        self.dataset.append((seq_features, clip_label))
                    elif self.sequence and self.crop:
                        seq_features = []
                        frame_features = []
                        num_players = clip_features.shape[0] // 9
                        for crop_idx in range(clip_features.shape[0]):
                            frame_features.append(clip_features[crop_idx])
                            if (crop_idx+1) % num_players == 0:
                                seq_features.append(frame_features)
                                frame_features = []
                        seq_features = np.array(seq_features) # shape [num_frames, num_players, 2048]
                        num_frames, num_players, _ = seq_features.shape
                        if num_players < 12:
                            pad  = np.zeros((num_frames, 12 - num_players, 2048))
                            seq_features = np.concatenate([seq_features, pad], axis=1)
                        self.dataset.append((seq_features, clip_label))
                    else: 
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