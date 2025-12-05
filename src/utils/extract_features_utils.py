"""
This script extracts deep features from volleyball player images using a fine-tuned ResNet-50 model.
It loads model checkpoints, applies preprocessing, and saves extracted features for each video clip.
"""
from Preprocessing import extract_features
from . import setup_logger, load_checkpoint
from stores import ModelProvidersFactory

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms


def prepare_model(logger, image_level: bool, num_classes: int, checkpoint_path: str, verbose=True):
    """
    Prepares the ResNet-50 model for feature extraction.

    Loads the model, applies preprocessing transforms, loads trained weights, and sets the model to evaluation mode.

    Returns:
        tuple: (model, transform) where model is the feature extractor and transform is the preprocessing pipeline.
    """
    if image_level:
        transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    # Check if a GPU is available if not, use a CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using Device:{device}")

    # Load ResNet-50 model with pretrained weights
    model_factory = ModelProvidersFactory()
    model = model_factory.create(model_name="b1", num_classes=num_classes)

    # Load a checkpoint saved during training
    logger.info("Loading the Model's Checkpoint...")
    checkpoint = torch.load(checkpoint_path)

    # Load trained weights into the model
    model = load_checkpoint(checkpoint=checkpoint, model=model)
    
    # Remove the classification head (i.e., the fully connected layers)
    model = nn.Sequential(*(list(model.children())[:-1]))

    # Send model to the device
    model.to(device)

    # Set the model to evaluation mode
    model.eval()
    logger.info(f"The Model is Ready and Sent to {device} Device, and Set to Eval Mode.")

    return model, transform



def extract(log_dir: str, videos_root: str,
            train_ids: list[int], val_ids: list[int],
            annot_root: str, output_root: str, num_classes: int,
            checkpoint_path: str, image_level: bool,
            image_classify: bool, verbose: bool):
    """
    function to extract features from volleyball video clips.

    Iterates through videos and clips, loads annotations, and saves extracted features as .npy files.
    """
    logger = setup_logger(
            log_file=__file__,
            log_dir=log_dir,
            log_to_console=verbose,
            use_tqdm=True
        )
    
    model, transform = prepare_model(
                            logger=logger,
                            image_level=image_level,
                            num_classes=num_classes,
                            checkpoint_path=checkpoint_path,
                            verbose=verbose
                        )

    videos_dirs = os.listdir(videos_root)
    videos_dirs.sort()

    # Iterate on each video and for each video iterate on each clip
    for idx, video_dir in enumerate(videos_dirs):
        if idx in train_ids or val_ids:
            logger.info(f"Working on Video Number: {idx}")
            video_dir_path = os.path.join(videos_root, video_dir)

            if not os.path.isdir(video_dir_path):
                continue

            clips_dir = os.listdir(video_dir_path)
            clips_dir.sort()

            for clip_dir in clips_dir:
                clip_dir_path = os.path.join(video_dir_path, clip_dir)

                if not os.path.isdir(clip_dir_path):
                    continue

                annot_file = os.path.join(annot_root, video_dir, clip_dir, f"{clip_dir}.txt")
                output_file = os.path.join(output_root, video_dir)

                if not os.path.isdir(output_file):
                    os.makedirs(output_file)
                
                output_file = os.path.join(output_file, f"{clip_dir}.npy")
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                extract_features(clip_dir_path, annot_file, output_file, model, transform, device, image_level, image_classify)
