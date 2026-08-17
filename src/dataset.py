from torchvision.io import decode_image
from torch.utils.data import Dataset
from src.configs import CLASS_TO_IDX
import pandas as pd
import torch

class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None,
                 target_transform=None):
        # annotations.csv -> pandas dataframe
        annot_df = pd.read_csv(annotations_file)

        # list of image file names
        self.img_filenames = list(annot_df["filename"].unique())

        # group ground truth bounded objects in images by their image file
        # names
        self.groups = annot_df.groupby("filename")

        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_filenames)

    def _create_target_tensor(self, img_filename):
        # group - all ground truth bounded objects in a single image
        group = self.groups.get_group(img_filename).drop(columns=["filename"])

        target_list = []
        for _, row in group.iterrows():
            target_list.append(list(row))

        return torch.tensor(target_list)

    def __getitem__(self, index):
        img_filename = self.img_filenames[index]

        img_path = self.img_dir / img_filename

        # Decode, Convert Datatype, and Normalize
        image = decode_image(img_path).to(torch.float32) / 255.0

        target_tensor = self._create_target_tensor(img_filename)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target_tensor = self.target_transform(target_tensor)

        return image, target_tensor