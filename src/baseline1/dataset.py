from torch.utils.data import Dataset
import os
from PIL import Image
from src.Preprocessing.volleyball_annot_loader import load_video_annot

class B1Dataset(Dataset):
    def __init__(self, videos_root: str, target_videos: list[int], transform=None):
        self.videos_root = videos_root
        self.target_videos = target_videos
        self.transform = transform

    def get_images(self):
        images = []
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
                            images.append(Image.open(img_path))
                            break  # Stop after finding the first matching image (there is only one per clip)
        return images
    
    def get_classes(self):
        classes = []
        videos_dirs = os.listdir(self.videos_root)
        videos_dirs.sort()

        for idx, video_dir in enumerate(videos_dirs):
            if idx in self.target_videos:
                video_dir_path = os.path.join(self.videos_root, video_dir)
                
                video_annot = os.path.join(video_dir_path, 'annotations.txt')
                clip_category_dict = load_video_annot(video_annot)

                # clips_dir = os.listdir(video_dir_path)
                # clips_dir.sort()

                for _, category in clip_category_dict.items():
                    classes.append(category)

        return classes


    def __len__(self):
        return len(self.get_images())

    def __getitem__(self, index):
        if index < 0 or index >= len(self.get_images()):
            raise IndexError("Index out of range")
        return self.get_images()[index], self.get_classes()[index]
    

if "__name__" == "__main__":
    dataset = B1Dataset(videos_root='/teamspace/studios/this_studio/Deep-Activity-Recognition/data/volleyball/volleyball_/videos', target_videos=[0, 1, 2])
    print(f"Number of images: {len(dataset)}")
    print(f"First image class: {dataset[0][1]}")