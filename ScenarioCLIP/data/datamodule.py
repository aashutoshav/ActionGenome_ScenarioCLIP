import torch
import lightning as L
from torch.utils.data import DataLoader
from .dataset import ActionGenomeDataset
import torchvision.transforms as T
import random
import json
import glob

class ActionGenomeDataModule(L.LightningDataModule):
    def __init__(self, metadata_dir, batch_size=8, num_workers=16):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preprocess = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])
        self.metadata_dir = metadata_dir
        self.metadata = []
        for file in glob.glob(f"{metadata_dir}/metadata*.json"):
            with open(file) as f:
                self.metadata.extend(json.load(f)['data'])
        with open(f"{metadata_dir}/classes.json") as f:
            self.classes = json.load(f)

    def collate(self, batch):
        image, caption, object_names, objects_cropped, relation_captions, relation_images = zip(*batch)
        images = torch.stack(image)
        return images, caption, object_names, objects_cropped, relation_captions, relation_images

    def setup(self, stage=None):
        random.shuffle(self.metadata)
        train_count = int(len(self.metadata)*.7)
        test_count = int(len(self.metadata)*.2)
        self.train_metadata = self.metadata[:train_count]
        self.val_metadata = self.metadata[train_count:-test_count]
        self.test_metadata = self.metadata[-test_count:]
        self.train_dataset = ActionGenomeDataset(transform=self.preprocess, metadata=self.train_metadata)
        self.val_dataset = ActionGenomeDataset(transform=self.preprocess, metadata=self.val_metadata)
        self.test_dataset = ActionGenomeDataset(transform=self.preprocess, metadata=self.test_metadata)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate, pin_memory=True)
