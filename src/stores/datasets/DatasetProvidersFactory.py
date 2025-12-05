from providers import FeaturesDataset, ImagesDataset

from torchvision import transforms
from torch.utils.data import DataLoader

class DatasetProvidersFactory:

    def __init__(self):
        pass

    def get_data_loaders(batch_size: int,
                        videos_root: str,
                        annot_root: str,
                        train_ids: list[int],
                        val_ids: list[int],
                        features: bool,
                        log_dir: str,
                        actions_dict: dict,
                        output_file: str = None,
                        image_level: bool = None,
                        crop: bool = None,
                        sequence: bool = None,
                        verbose: bool = False):
        """Creates train and val data loaders.
        Returns:
            tuple: (trainloader, valloader)
        """

        if features:
            train_dataset = FeaturesDataset(
                output_file=output_file,
                videos_root=videos_root,
                target_videos=train_ids,
                categories_dict=actions_dict,
                log_dir=log_dir,
                crop=crop,
                sequence=sequence,
                verbose=verbose
            )
            val_dataset = FeaturesDataset(
                output_file=output_file,
                videos_root=videos_root,
                target_videos=val_ids,
                categories_dict=actions_dict,
                log_dir=log_dir,
                crop=crop,
                sequence=sequence,
                verbose=verbose
            )
        else:
            if image_level:
                transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

                train_dataset = ImagesDataset(videos_root=videos_root,
                                    target_videos=train_ids,
                                    annot_root=annot_root,
                                    log_dir=log_dir,
                                    image_level=image_level,
                                    actions_dict=actions_dict,
                                    transform=transform,
                                    verbose=verbose
                                    )
                val_dataset = ImagesDataset(videos_root=videos_root,
                                    target_videos=val_ids,
                                    annot_root=annot_root,
                                    log_dir=log_dir,
                                    image_level=image_level,
                                    actions_dict=actions_dict,
                                    transform=transform,
                                    verbose=verbose
                                    )
            else:
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                train_dataset = ImagesDataset(videos_root=videos_root,
                                    target_videos=train_ids,
                                    annot_root=annot_root,
                                    log_dir=log_dir,
                                    image_level=image_level,
                                    actions_dict=actions_dict,
                                    transform=transform,
                                    verbose=verbose
                                    )
                val_dataset = ImagesDataset(videos_root=videos_root,
                                    target_videos=val_ids,
                                    annot_root=annot_root,
                                    log_dir=log_dir,
                                    image_level=image_level,
                                    actions_dict=actions_dict,
                                    transform=transform,
                                    verbose=verbose
                                    )

        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return trainloader, valloader