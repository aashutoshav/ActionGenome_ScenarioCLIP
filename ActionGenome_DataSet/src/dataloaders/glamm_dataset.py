import torch
import os
import json
from PIL import Image, ImageOps
from typing import Tuple
from torchvision import transforms
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def custom_collate_glamm(batch):
    img_files, images, annotations= zip(*batch)
    return list(img_files), list(images), list(annotations)

def custom_collate_glammwLLaVa(batch):
    img_files, images, annotations, llava_annots = zip(*batch)
    return list(img_files), list(images), list(annotations), list(llava_annots)

def custom_collate_glammwDino(batch):
    img_files, images, annotations, dino_annots = zip(*batch)
    return list(img_files), list(images), list(annotations), list(dino_annots)


def resize_with_zero_padding(image, new_size):
    return ImageOps.pad(image, new_size, method=Image.NEAREST, color=0, centering=(0.5, 0.5))

class GLAMM_Dataset(Dataset):
    def __init__(self, root_dir, start_chunk=0, end_chunk=999, image_shape=(512, 512), transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and JSON files.
            start_chunk (int): Index of the Starting Chunk
            end_chunk (int): Index of the Ending Chunk
            image_shape (tuple): Desired size of the images
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.image_shape = image_shape
        self.transform = transform or transforms.Compose([
            transforms.Resize(image_shape),
            transforms.ToTensor(),
        ])
        # self.chunk_list = [f"sa_{i:06d}" for i in range(start_chunk, end_chunk + 1)]
        self.chunk_list = [f"sa_000108", f'sa_000029']
        self.image_data = []

        # Collecting all paths using ThreadPoolExecutor for efficiency
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            features = []
            for chunk in self.chunk_list:
                chunk_path = os.path.join(root_dir, chunk)
                if os.path.exists(chunk_path):
                    features.extend([executor.submit(self.collect_files, chunk_path, file) 
                                    for file in os.listdir(chunk_path) if file.endswith('.jpg')])

            for feature in tqdm(features, total=len(features), desc="Loading image paths"):
                result = feature.result()
                if result:
                    self.image_data.append(result)

        print(f"Found {len(self.image_data)} Files.")
        print(f"Image Data Example: {self.image_data[0]}")

    def collect_files(self, chunk_path, file):
        img_file = os.path.join(chunk_path, file)
        json_file = img_file.replace('.jpg', '.json')
        if os.path.exists(json_file):
            return img_file, json_file
        return None

    def load_image_data(self, img_json_pair):
        img_file, json_file = img_json_pair
        image = Image.open(img_file).convert('RGB')
        image = resize_with_zero_padding(image, self.image_shape)
        image = self.transform(image)

        with open(json_file, 'r') as jfile:
            annotations = json.load(jfile)

        return img_file, image, annotations

    def __len__(self) -> int:
        """Return the total number of image files."""
        return len(self.image_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        """Generates one sample of data."""
        img_file, image, annotations = self.load_image_data(self.image_data[idx])
        return img_file, image, annotations

class GLAMMwLLaVAResp_Dataset(Dataset):
    def __init__(self, root_dir, start_chunk=0, end_chunk=999, image_shape=(512, 512), transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and JSON files.
            start_chunk (int): Index of the Starting Chunk
            end_chunk (int): Index of the Ending Chunk
            image_shape (tuple): Desired size of the images
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.image_shape = image_shape
        self.transform = transform or transforms.Compose([
            transforms.Resize(image_shape),
            transforms.ToTensor(),
        ])
        # self.chunk_list = [f"sa_{i:06d}" for i in range(start_chunk, end_chunk + 1)]
        self.chunk_list = [f"sa_000029", f"sa_000108"]
        self.image_data = []
        
        print("Loading LLaVaResp Dataset")

        # Collecting all paths using ThreadPoolExecutor for efficiency
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            features = []
            for chunk in self.chunk_list:
                chunk_path = os.path.join(root_dir, chunk)
                print(f'Chunk Path: {chunk_path}')
                if os.path.exists(chunk_path):
                    features.extend([executor.submit(self.collect_files, chunk_path, file) 
                                    for file in os.listdir(chunk_path) if file.endswith('.jpg')])

            for feature in tqdm(features, total=len(features), desc="Loading image paths"):
                result = feature.result()
                if result:
                    self.image_data.append(result)

        print(f"Found {len(self.image_data)} Files.")

    def collect_files(self, chunk_path, file):
        img_file = os.path.join(chunk_path, file)
        json_file = img_file.replace('.jpg', '.json')
        llavaResp_file = img_file.replace('.jpg', '_llava_next.json')
        
        if os.path.exists(json_file) and os.path.exists(llavaResp_file):
            return img_file, json_file, llavaResp_file
        return None

    def load_image_data(self, img_json_pair):
        img_file, json_file, llavaResp_file = img_json_pair
        image = Image.open(img_file).convert('RGB')
        image = resize_with_zero_padding(image, self.image_shape)
        image = self.transform(image)
        if image.max()<=1:
            image=(1-image)*255 # NOTE: Very important to convert to 0-255 scale for grounding-dino, else no result
            image=image.int() # NOTE: Throws an error otherwise
 
        with open(json_file, 'r') as jfile:
            prev_annotations = json.load(jfile)
        
        with open(llavaResp_file, 'r') as jfile:
            llava_annotations = json.load(jfile)

        return img_file, image, prev_annotations, llava_annotations

    def __len__(self) -> int:
        """Return the total number of image files."""
        return len(self.image_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        """Generates one sample of data."""
        img_file, image, annotations, llave_annotation = self.load_image_data(self.image_data[idx])
        return img_file, image, annotations, llave_annotation


class GLaMMwDinoResp_Dataset(Dataset):
    def __init__(self, root_dir, start_chunk=0, end_chunk=999, image_shape=(512, 512), transform=None):
        super().__init__()

        """
        Args:
            root_dir (string): Directory with all the images and JSON files.
            start_chunk (int): Index of the Starting Chunk
            end_chunk (int): Index of the Ending Chunk
            image_shape (tuple): Desired size of the images
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.image_shape = image_shape
        self.transform = transform or transforms.Compose([
            transforms.Resize(image_shape),
            transforms.ToTensor(),
        ])
        # self.chunk_list = [f"sa_{i:06d}" for i in range(start_chunk, end_chunk + 1)]
        self.image_data = []
        self.chunk_list = [f"sa_000029", f"sa_000108"]
        
        print(f"Loading DinoResp Dataset")

        # Collecting all paths using ThreadPoolExecutor for efficiency
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            features = []
            for chunk in self.chunk_list:
                chunk_path = os.path.join(root_dir, chunk)
                if os.path.exists(chunk_path):
                    features.extend([executor.submit(self.collect_files, chunk_path, file) 
                                    for file in os.listdir(chunk_path) if file.endswith('.jpg') and 'visualised' not in file])

            for feature in tqdm(features, total=len(features), desc="Loading image paths"):
                result = feature.result()
                if result:
                    self.image_data.append(result)

        print(f"Found {len(self.image_data)} Files.")

    def collect_files(self, chunk_path, file):
        img_file = os.path.join(chunk_path, file)
        json_file = img_file.replace('.jpg', '.json')
        dinoResp_file = img_file.replace('.jpg', '_grounding_dino.json')

        if os.path.exists(json_file) and os.path.exists(dinoResp_file):
            return img_file, json_file, dinoResp_file
        return None

    def load_image_data(self, img_json_pair):
        img_file, json_file, dinoResp_file = img_json_pair
        image = Image.open(img_file).convert('RGB')
        image = resize_with_zero_padding(image, self.image_shape)
        image = self.transform(image)
        if image.max()<=1:
            image=(1-image)*255 # NOTE: Very important to convert to 0-255 scale for grounding-dino, else no result
            image=image.int() # NOTE: Throws an error otherwise

        with open(json_file, 'r') as jfile:
            prev_annotations = json.load(jfile)

        with open(dinoResp_file, 'r') as jfile:
            dino_annotations = json.load(jfile)

        return img_file, image, prev_annotations, dino_annotations

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        """Generates one sample of data."""
        img_file, image, annotations, dino_annotations = self.load_image_data(self.image_data[idx])
        return img_file, image, annotations, dino_annotations

    def __len__(self) -> int:
        """Return the total number of image files."""
        return len(self.image_data)
