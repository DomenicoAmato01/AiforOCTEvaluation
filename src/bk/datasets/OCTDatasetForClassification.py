import pandas as pd
import torch
from torch.utils.data import Dataset
import os
from skimage import io
import numpy as np

class OCTDatasetForClassification(Dataset):
    """OCT Dataset for classification task."""

    def __init__(self, excel_file, root_dir, transform=None, train=True):
        """
        Arguments:
            excel_file (string): Path to the excel file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.oct_annotations = pd.read_excel(excel_file)
        if train:
            self.oct_annotations = self.oct_annotations[self.oct_annotations["set"] == "train"].reset_index()
        else:
            self.oct_annotations = self.oct_annotations[self.oct_annotations["set"] == "test"].reset_index()
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.oct_annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        fn = self.oct_annotations.loc[idx, "nomefile"]

        patient = fn.split('_')[0]
        img_name = os.path.join(self.root_dir, patient, "images", fn)
        image = io.imread(img_name)/255.0
        image = torch.tensor(image, dtype=torch.float32)
        image = torch.stack([image, image, image], dim=0)
        image = torch.unsqueeze(image, 0)
        # image = torch.stack([image, image, image], dim=0)
        # image = torch.unsqueeze(image, 0)
        # # image = image[np.newaxis, ...]  # Add channel dimension
        labels = self.oct_annotations.loc[idx, "c"]
        if self.transform:
            image = self.transform(image)
        image = torch.squeeze(image, 0)
        sample = {'image': image, 'label': labels}

        

        return sample