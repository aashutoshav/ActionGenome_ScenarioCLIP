import os
import subprocess
from typing import Tuple
import torch
from PIL import Image
from torchvision import transforms
import json
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def custom_collate_kinetics(batch):
    img_files, images, annotations = zip(*batch)
    return list(img_files), list(images), list(annotations)

def custom_collate_kineticswLLaVa(batch):
    img_files, images, annotations, llava_annots = zip(*batch)
    return list(img_files), list(images), list(annotations), list(llava_annots)


def custom_collate_kineticswDino(batch):
    img_files, images, annotations, dino_annots = zip(*batch)
    return list(img_files), list(images), list(annotations), list(dino_annots)


class Kinetics_Dataset(Dataset):
    def __init__(
        self,
        root_dir,
        storage_dir,
        start_chunk,
        end_chunk,
        image_shape=None,
        transform=None,
    ):
        self.root_dir = root_dir
        self.image_shape = image_shape
        self.transform = transform or transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.storage_dir = storage_dir
        self.start_chunk = start_chunk
        self.end_chunk = end_chunk

        self.video_folder = os.path.join(self.root_dir, "val_256")
        print(f'Video Folder: {self.video_folder}')

        self.all_folders = [
            f
            for f in os.listdir(self.video_folder)
            if os.path.isdir(os.path.join(self.video_folder, f))
        ]
        self.all_folders = sorted(self.all_folders)

        self.chunk_list = self.all_folders[
            self.start_chunk : min(len(self.all_folders) - 1, self.end_chunk + 1)
        ]

        self.image_data = []

        self.load_image_data_in_chunks()

        print(f"Found {len(self.image_data)} files.")
        print(f"Image Data Ex: {self.image_data[0]}")

    def load_image_data_in_chunks(self):
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []

            for folder in self.chunk_list:
                folder_path = os.path.join(self.video_folder, folder)
                print(f"Folder Path: {folder_path}")
                if os.path.exists(folder_path) and folder in self.chunk_list and os.path.isdir(folder_path):
                    for sub_folder in os.listdir(folder_path):
                        if os.path.exists(os.path.join(folder_path, sub_folder)) and os.path.isdir(os.path.join(folder_path, sub_folder)):
                            sub_folder_path = os.path.join(folder_path, sub_folder)
                            print(f'Sub Folder Path: {sub_folder_path}')
                            futures.extend(
                                [
                                    executor.submit(
                                        self.collect_files, folder, folder_path, sub_folder_path, filename
                                    )
                                    for filename in sorted(os.listdir(sub_folder_path))[:1]
                                    if filename.endswith(".jpg") or filename.endswith(".png")
                                ]
                            )

            for future in tqdm(futures, total=len(futures), desc="Loading image paths"):
                result = future.result()
                if result:
                    self.image_data.append(result)

    def collect_files(self, folder, folder_path, sub_folder_path, filename):
        img_file = os.path.join(sub_folder_path, filename)
        scenario_action_annot = os.path.basename(folder_path)

        if os.path.exists(img_file):
            return img_file, scenario_action_annot
        return None

    def load_image_data(self, img_annot_pair):
        img_file, annot_class = img_annot_pair
        image = Image.open(img_file).convert("RGB")

        return img_file, image, annot_class

    def __len__(self) -> int:
        """Return the total number of image files."""
        return len(self.image_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        """Generates one sample of data."""
        img_file, image, scenarioAction_class = self.load_image_data(
            self.image_data[idx]
        )
        return img_file, image, scenarioAction_class


def is_json_file_empty(file_path):
    """Check if the given JSON file is empty."""
    try:
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            with open(file_path, "r") as f:
                first_char = f.read(1)
                if not first_char:
                    print(f"{file_path} is empty.")
                    return True
        else:
            print(f"{file_path} is either missing or empty.")
            return True
    except Exception as e:
        print(f"Error checking file {file_path}: {e}")
        return True

    return False


class KineticswLLaVAResp_Dataset(Dataset):
    def __init__(
        self,
        root_dir,
        storage_dir,
        start_chunk,
        end_chunk,
        image_shape=None,
        transform=None,
    ):
        self.root_dir = root_dir
        self.image_shape = image_shape
        self.transform = transform or transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.storage_dir = storage_dir
        self.start_chunk = start_chunk
        self.end_chunk = end_chunk

        self.video_folder = os.path.join(self.root_dir, "val_256")

        self.all_folders = [
            f
            for f in os.listdir(self.video_folder)
            if os.path.isdir(os.path.join(self.video_folder, f))
        ]
        self.all_folders = sorted(self.all_folders)

        self.chunk_list = self.all_folders[
            self.start_chunk : min(len(self.all_folders) - 1, self.end_chunk + 1)
        ]

        self.image_data = []

        self.load_image_data_in_chunks()

        print(f"Found {len(self.image_data)} files.")
        print(f"Image Data Ex: {self.image_data[0]}")

    def load_image_data_in_chunks(self):
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []

            for folder in self.chunk_list:
                folder_path = os.path.join(self.video_folder, folder)
                if (
                    os.path.exists(folder_path)
                    and folder in self.chunk_list
                    and os.path.isdir(folder_path)
                ):
                    for sub_folder in os.listdir(folder_path):
                        if os.path.exists(
                            os.path.join(folder_path, sub_folder)
                        ) and os.path.isdir(os.path.join(folder_path, sub_folder)):
                            sub_folder_path = os.path.join(folder_path, sub_folder)
                            futures.extend(
                                [
                                    executor.submit(
                                        self.collect_files,
                                        folder,
                                        folder_path,
                                        sub_folder_path,
                                        filename,
                                    )
                                    for filename in sorted(os.listdir(sub_folder_path))[:1]
                                    if filename.endswith(".jpg")
                                    or filename.endswith(".png")
                                ]
                            )

            for future in tqdm(futures, total=len(futures), desc="Loading image paths"):
                result = future.result()
                if result:
                    self.image_data.append(result)

    def collect_files(self, folder, folder_path, sub_folder_path, filename):
        img_file = os.path.join(sub_folder_path, filename)
        gemmaResp_file = os.path.join(
            self.storage_dir,
            "gemma_jsons",
            folder,
            os.path.basename(sub_folder_path),
            filename.replace(".jpg", "_gemma.json"),
        )
        scenario_action_annot = os.path.basename(folder_path)

        if os.path.exists(img_file) and os.path.exists(gemmaResp_file):
            try:
                with Image.open(img_file) as img:
                    img.verify()

                if os.path.getsize(gemmaResp_file) > 0:
                    return img_file, scenario_action_annot, gemmaResp_file

            except Exception as e:
                pass
        return None

    def load_image_data(self, img_annot_pair):
        img_file, annot_class, gemmaResp_file = img_annot_pair
        image = Image.open(img_file).convert("RGB")
        image = self.transform(image)
        if image.max() <= 1:
            image = (1 - image) * 255
            image = image.int()

        with open(gemmaResp_file, "r") as jfile:
            gemmaAnnotations = json.load(jfile)

        return img_file, image, annot_class, gemmaAnnotations

    def __len__(self) -> int:
        """Return the total number of image files."""
        return len(self.image_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        """Generates one sample of data."""
        img_file, image, scenarioAction_class, gemma_annotations = self.load_image_data(
            self.image_data[idx]
        )
        return img_file, image, scenarioAction_class, gemma_annotations


class KineticswDinoResp_Dataset(Dataset):
    def __init__(
        self,
        root_dir,
        storage_dir,
        start_chunk,
        end_chunk,
        image_shape=None,
        transform=None,
    ):
        self.root_dir = root_dir
        self.image_shape = image_shape
        self.transform = transform or transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.storage_dir = storage_dir
        self.start_chunk = start_chunk
        self.end_chunk = end_chunk

        self.video_folder = os.path.join(self.root_dir, "val_256")

        self.all_folders = [
            f
            for f in os.listdir(self.video_folder)
            if os.path.isdir(os.path.join(self.video_folder, f))
        ]
        self.all_folders = sorted(self.all_folders)

        self.chunk_list = self.all_folders[
            self.start_chunk : min(len(self.all_folders) - 1, self.end_chunk + 1)
        ]

        self.image_data = []

        self.load_image_data_in_chunks()

        print(f"Found {len(self.image_data)} files.")
        print(f"Image Data Ex: {self.image_data[0]}")

    def load_image_data_in_chunks(self):
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []

            for folder in self.chunk_list:
                folder_path = os.path.join(self.video_folder, folder)
                if (
                    os.path.exists(folder_path)
                    and folder in self.chunk_list
                    and os.path.isdir(folder_path)
                ):
                    for sub_folder in os.listdir(folder_path):
                        if os.path.exists(
                            os.path.join(folder_path, sub_folder)
                        ) and os.path.isdir(os.path.join(folder_path, sub_folder)):
                            sub_folder_path = os.path.join(folder_path, sub_folder)
                            futures.extend(
                                [
                                    executor.submit(
                                        self.collect_files,
                                        folder,
                                        folder_path,
                                        sub_folder_path,
                                        filename,
                                    )
                                    for filename in sorted(os.listdir(sub_folder_path))[:1]
                                    if filename.endswith(".jpg")
                                    or filename.endswith(".png")
                                ]
                            )

            for future in tqdm(futures, total=len(futures), desc="Loading image paths"):
                result = future.result()
                if result:
                    self.image_data.append(result)

    def collect_files(self, folder, folder_path, sub_folder_path, filename):
        img_file = os.path.join(sub_folder_path, filename)
        gemmaResp_file = os.path.join(
            self.storage_dir,
            "gemma_jsons",
            folder,
            os.path.basename(sub_folder_path),
            filename.replace(".jpg", "_gemma.json"),
        )
        dinoResp_file = os.path.join(
            self.storage_dir,
            "dino_results",
            folder,
            os.path.basename(sub_folder_path),
            filename.replace('.jpg', '_grounding_dino.json')
        )
        scenario_action_annot = os.path.basename(folder_path)

        if os.path.exists(img_file) and os.path.exists(gemmaResp_file) and os.path.exists(dinoResp_file):
            try:
                with Image.open(img_file) as img:
                    img.verify()

                if os.path.getsize(gemmaResp_file) > 0 and os.path.getsize(dinoResp_file):
                    return img_file, gemmaResp_file, dinoResp_file

            except Exception as e:
                pass
        return None

    def load_image_data(self, img_json_triple):
        img_file, json_file, dinoResp_file = img_json_triple
        image = Image.open(img_file).convert("RGB")
        image = self.transform(image)
        if image.max() <= 1:
            image = (1 - image) * 255
            image = image.int()

        with open(json_file, "r") as jfile:
            gemma_annotations = json.load(jfile)

        with open(dinoResp_file, "r") as jfile:
            dino_annotations = json.load(jfile)

        return img_file, image, gemma_annotations, dino_annotations

    def __len__(self) -> int:
        """Return the total number of image files."""
        return len(self.image_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        """Generates one sample of data."""
        img_file, image, gemma_annotations, dino_annotations = self.load_image_data(
            self.image_data[idx]
        )
        return img_file, image, gemma_annotations, dino_annotations
