from stores.models.providers import *
from stores.models.ModelsEnums import ModelsEnums

import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ModelProvidersFactory:

    def __init__(self):
        pass

    def create(self, model_name: str, num_classes: int,
               input_size: int = None, hidden_size1: int = None, hidden_size2: int = None,
               num_layers: int = None, log_dir: str = None, verbose: bool=True):
        
        if model_name == ModelsEnums.ACTIVITY_CLASSIFIER.value:
            original_model = resnet50(weights=ResNet50_Weights.DEFAULT, progress=verbose)
            layers = list(original_model.children())[:-1]
            truncated_model = nn.Sequential(*layers)
            model = Activity_Classifier(truncated_model, num_classes, log_dir, verbose)
        
        elif model_name == ModelsEnums.GROUP_ACTIVITY_CLASSIFIER.value:
            model = Group_Activity_Classifier(input_size, num_classes, log_dir, verbose)
        
        elif model_name == ModelsEnums.Group_Activity_Temporal_Classifier.value:
            model = Group_Activity_Temporal_Classifier(num_classes, input_size, hidden_size1, num_layers, log_dir, verbose)

        elif model_name == ModelsEnums.Pooled_Players_Activity_Temporal_Classifier.value:
            model = Pooled_Players_Activity_Temporal_Classifier(
                num_classes=num_classes,
                input_size=input_size,
                hidden_size=hidden_size1,
                num_layers=num_layers,
                log_dir=log_dir,
                verbose=verbose
            )
        
        elif model_name == ModelsEnums.Two_Stage_Activity_Temporal_Classifier.value:
            model = Two_Stage_Activity_Temporal_Classifier(
                num_classes=num_classes,
                input_size=input_size,
                hidden_size1=hidden_size1,
                hidden_size2=hidden_size2,
                num_layers=num_layers,
                log_dir=log_dir,
                verbose=verbose
            )
        
        elif model_name == ModelsEnums.Two_Stage_Pooled_Teams_Activity_Temporal_Classifier.value:
            model = Two_Stage_Pooled_Teams_Activity_Temporal_Classifier(
                num_classes=num_classes,
                input_size=input_size,
                hidden_size1=hidden_size1,
                hidden_size2=hidden_size2,
                num_layers=num_layers,
                log_dir=log_dir,
                verbose=verbose
            )

        return model
        
