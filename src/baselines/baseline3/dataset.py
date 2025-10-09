from src.utils.logging_utils import setup_logger
from src.utils.config_utils import load_config
from src.Preprocessing.volleyball_annot_loader import load_tracking_annot

from torch.utils.data import Dataset
from PIL import Image
import os

CONFIG = load_config()

class B3Dataset(Dataset):
    def __init__(self, videos_root: str, target_videos: list[int], annot_root, transform=None, verbose=False):
        self.videos_root = videos_root
        self.target_videos = target_videos
        self.annot_root = annot_root
        self.verbose = verbose
        self.tranform = transform
        self.logger = setup_logger(
            log_file=__file__,
            log_dir="",
            log_to_console=self.verbose,
            use_tqdm=True
        )
        self.logger.info("B3Dataset Module Initialized Successfully!")
    

    def get_images_paths_labels(self):
        self.logger.info("Collecting Images' paths and labels...")

        self.dataset = []
        videos_dirs = os.listdir(self.videos_root)
        videos_dirs.sort()

        # Iterate on each video and for each video iterate on each clip
        for idx, video_dir in enumerate(videos_dirs):
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
                            self.dataset.append((cropped_image, box_info.category))

                    except Exception as e:
                        print(f"An error occurred: {e}")


    def __len__(self):
        return len(self.dataset)
    

    def __getitem__(self, index):
        if index > 0 or index >= len(self.dataset):
            raise IndexError("Index out of range")
        img, label = self.dataset[index]

        if self.transform:
            img = self.tranform(img)
        return img, label
    

if __name__ == "__main__":
    dataset = B3Dataset(videos_root="", target_videos=[], annot_root="")
        