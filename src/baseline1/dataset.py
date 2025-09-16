from torch.utils.data import Dataset
import os
from PIL import Image
from src.Preprocessing.volleyball_annot_loader import load_video_annot
from src.utils.config_utils.load_config import load_config
from src.utils.stream_utils.stream_utils import log_stream

CONFIG = load_config()
log_stream("Dataset_logs", prog="baselines_logs/baseline1_logs")

class B1Dataset(Dataset):
    """Custom Dataset for loading volleyball video frames and labels.

    Attributes:
        videos_root (str): Root directory containing video folders.
        target_videos (list[int]): List of video indices to include.
        transform (callable, optional): Transform to apply to images.
        dataset (list): List of (image_path, label) tuples.
        verbose (bool): If True, prints info logs.
    """

    def __init__(self, videos_root: str, target_videos: list[int], transform=None):
        """
        Args:
            videos_root (str): Root directory containing video folders.
            target_videos (list[int]): List of video indices to include.
            transform (callable, optional): Transform to apply to images.
        """
        self.videos_root = videos_root
        self.target_videos = target_videos
        self.transform = transform
        self.verbose = CONFIG.get("verbose", False)

        if self.verbose:
            print("[INFO] Initializing B1Dataset...")
        self.get_images_paths_labels()
        if self.verbose:
            print(f"[INFO] B1Dataset initialized with {len(self.dataset)} samples.")

    def get_images_paths_labels(self):
        """Populates self.dataset with image paths and labels."""
        if self.verbose:
            print("[INFO] Collecting image paths and labels...")
        self.dataset = []
        videos_dirs = os.listdir(self.videos_root)
        videos_dirs.sort()

        # Iterate on each video, in each video: iterate on each clip, in each clip: iterate on each image, append it and its category to dataset list
        for idx, video_dir in enumerate(videos_dirs):
            if idx in self.target_videos:
                video_dir_path = os.path.join(self.videos_root, video_dir)

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

                    for img in os.listdir(clip_dir_path):
                        img_path = os.path.join(clip_dir_path, img)
                        label = CONFIG["CATEGORIES_DICT"][clip_category_dict[clip_dir]]
                        self.dataset.append((img_path, label))
        if self.verbose:
            print(f"[INFO] Collected {len(self.dataset)} image-label pairs.")

    def __len__(self):
        """Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.dataset)

    def __getitem__(self, index):
        """Retrieves the image and label at the specified index.

        Args:
            index (int): Index of the sample.

        Returns:
            tuple: (image, label)
        """
        if index < 0 or index >= len(self.dataset):
            raise IndexError("Index out of range")
        image_path, label = self.dataset[index]

        if self.verbose:
            print(f"[INFO] Loading image at index {index}: {image_path}")
        img = Image.open(image_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

if __name__ == "__main__":
    dataset = B1Dataset(videos_root=CONFIG["PATH"]["videos_root"], target_videos=[0])