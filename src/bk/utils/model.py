from sklearn.metrics import accuracy_score
import yaml
import torch
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from sklearn.metrics import confusion_matrix
from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig

def get_transform(size):
    _transf = Compose([
            #ToTensor(),
            Resize((size['width'], size['height'])),

        #     Resize(size['height']),
        #     ToTensor(),
            # Normalize(mean=[0.5], std=[0.5])
        ]) 
    return _transf

def collate_fn(examples):
    pixel_values = torch.stack([example['image'] for example in examples])
    labels = torch.tensor([int(example['label']) for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def compute_metrics(p):
        """Computes accuracy on a batch of predictions"""
        preds = np.argmax(p.predictions, axis=1)
        
        cm = confusion_matrix(p.label_ids, preds).tolist()

        return {"accuracy": accuracy_score(p.label_ids, preds), "confusion_matrix": cm}

def get_model_config(model_name, task="classification"):
    with open("config/models.yml", "r") as f:
        model = yaml.safe_load(f)
    return model.get(task, {}).get(model_name, {})

def get_classification_model(param_name):
    with open("config/main.yml", "r") as f:
        config = yaml.safe_load(f)
    
    model_name = config.get(param_name, {}).get("class-model", "")
    model_conf = get_model_config(model_name)
    
    VitConf = ViTConfig(**model_conf, num_labels=config.get(param_name, {}).get("num_classes", 2))
    processor = ViTImageProcessor.from_pretrained(f'google/{model_name}', do_convert_rgb=True)
    model = ViTForImageClassification.from_pretrained(f'google/{model_name}', config=VitConf, ignore_mismatched_sizes=True)

    return processor, model, model_name

def get_segmentation_model(param_name):
    with open("config/main.yml", "r") as f:
        config = yaml.safe_load(f)
    
    model_name = config.get(param_name, {}).get("seg-model", "")
    print(f"Segmentation model selected: {model_name}")
    if model_name == "U-Net":
        from monai.networks.nets import UNet

        model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2,2,2,2),
            num_res_units=2,
        )
    elif model_name == "AttentionU-Net":
        from monai.networks.nets import AttentionUnet

        model = AttentionUnet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2,2,2,2),
        )
    elif model_name == "U-Net++":
        from monai.networks.nets import BasicUNetPlusPlus

        model = BasicUNetPlusPlus(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            features=(16, 32, 64, 128, 256, 32),
        )
    else:
        raise ValueError(f"Model {model_name} not supported for segmentation.")

    return model, model_name