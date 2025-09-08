from torch.utils.data import Dataset
import os
from PIL import Image
from src.Preprocessing.volleyball_annot_loader import load_video_annot
from src.utils.config_utils.load_config import load_config

CONFIG = load_config()

class B1Dataset(Dataset):
    def __init__(self, videos_root: str, target_videos: list[int], transform=None):
        self.videos_root = videos_root
        self.target_videos = target_videos
        self.transform = transform

        self.get_images_paths_labels()

    def get_images_paths_labels(self):
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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if index < 0 or index >= len(self.dataset):
            raise IndexError("Index out of range")
        image_path, label = self.dataset[index]

        img = Image.open(image_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        
        return img, label
    

if __name__ == "__main__":
    dataset = B1Dataset(videos_root=CONFIG["PATH"]["videos_root"], target_videos=[0])