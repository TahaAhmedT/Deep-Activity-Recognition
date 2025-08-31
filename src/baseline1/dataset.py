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

    def get_images(self):
        images_dict = {}
        videos_dirs = os.listdir(self.videos_root)
        videos_dirs.sort()

        # Iterate on each video, in each video: iterate on each clip, in each clip: iterate on each image untill find the image that has the same id like the clip
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

                    for img in os.listdir(clip_dir_path):
                        img_path = os.path.join(clip_dir_path, img)
                        if img.endswith('.jpg') and img[:-4] == clip_dir:
                            images_dict[clip_dir] = img_path
                            break  # Stop after finding the first matching image (there is only one per clip)
        return images_dict

    def get_classes(self):
        classes_dict = {}
        videos_dirs = os.listdir(self.videos_root)
        videos_dirs.sort()

        for idx, video_dir in enumerate(videos_dirs):
            if idx in self.target_videos:
                video_dir_path = os.path.join(self.videos_root, video_dir)
                
                video_annot = os.path.join(video_dir_path, 'annotations.txt')
                clip_category_dict = load_video_annot(video_annot)

                for clip, category in clip_category_dict.items():
                    classes_dict[clip] = category

        return classes_dict

    def print_info(self):
        print(f"Number of Images: {len(self.get_images())}")
        print(f"Number of Classes: {len(self.get_classes())}")

    def prepare_data(self):
        # 1. Merge two dicts by keys (images, classes)
        # Sort by keys to keep consistent order
        keys = sorted(self.get_images().keys())

        self.images = [self.get_images()[k] for k in keys]
        self.labels = [self.get_classes()[k] for k in keys]

        print("Images:", self.images)
        print("Labels:", self.labels)

        # 2. Convert labels to numeric
        categories_dict = CONFIG["CATEGORIES_DICT"]
        self.labels_numeric = [categories_dict[label] for label in self.labels]

        print("Numeric labels:", self.labels_numeric)


    def __len__(self):
        return len(self.get_images())

    def __getitem__(self, index):
        if index < 0 or index >= len(self.get_images()):
            raise IndexError("Index out of range")
        image_path = self.images[index]
        label = self.labels_numeric[index]

        img = Image.open(image_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        
        return img, label
    

if __name__ == "__main__":
    dataset = B1Dataset(videos_root=CONFIG["PATH"]["videos_root"], target_videos=[0, 1, 2])
    # dataset.print_info()
    dataset.prepare_data()
    # print(f"First image class: {dataset[0][1]}")