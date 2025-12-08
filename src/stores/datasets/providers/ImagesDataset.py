from utils.logging_utils import setup_logger
from Preprocessing.volleyball_annot_loader import load_tracking_annot, load_video_annot

from torch.utils.data import Dataset
from PIL import Image
import os


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
                            label = self.actions_dict[clip_category_dict[clip_dir]]
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
    
