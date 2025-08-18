from torch.utils.data import Dataset
import os
from PIL import Image

class B1Dataset(Dataset):
    def __init__(self, videos_root: str, target_dirs: list[int], transform=None):
        self.videos_root = videos_root
        self.target_dirs = target_dirs
        self.transform = transform

    def get_images(self):
        images = []
        videos_dirs = os.listdir(self.videos_root)
        videos_dirs.sort()

        # Iterate on each video, in each video: iterate on each clip, in each clip: iterate on each image untill find the image that has the same id like the clip
        for idx, video_dir in enumerate(videos_dirs):
            if idx in self.target_dirs:
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
                            images.append(Image.open(img_path))
                            break  # Stop after finding the first matching image (there is only one per clip)
        return images

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass