import torch
import os
import glob
import json
from PIL import Image, ImageOps
from typing import Tuple
from torchvision import transforms
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def custom_collate_CC3M(batch):
    img_files, images, annotations = zip(*batch)
    return list(img_files), list(images), list(annotations)


def custom_collate_CC3MwLLaVa(batch):
    img_files, images, annotations, llava_annots = zip(*batch)
    return list(img_files), list(images), list(annotations), list(llava_annots)


def custom_collate_CC3MwDino(batch):
    img_files, images, annotations, dino_annots = zip(*batch)
    return list(img_files), list(images), list(annotations), list(dino_annots)


def resize_with_zero_padding(image, new_size):
    return ImageOps.pad(
        image, new_size, method=Image.NEAREST, color=0, centering=(0.5, 0.5)
    )


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


class CC3M_Dataset(Dataset):
    def __init__(
        self,
        root_dir,
        start_chunk,
        end_chunk,
        image_shape=(512, 512),
        storage_dir="",
        transform=None,
    ):
        """
        Chunks: 0 to n inclusive
        """
        self.root_dir = root_dir
        self.image_shape = image_shape
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize(image_shape),
                transforms.ToTensor(),
            ]
        )
        self.storage_dir = storage_dir

        self.chunk_list = [x for x in range(332)]
        self.chunk_list = [str(y).zfill(5) for y in self.chunk_list][start_chunk : end_chunk + 1]

        self.image_data = []

        self.classes_path = (
            "/home/zeta/Workbenches/ActionGenome_DataSet/cc3m_dataset/classes.json"
        )

        self.action_list = self.load_action_list_from_metadata()

        self.load_image_data_in_chunks()

        print(f"Found {len(self.image_data)} Files.")
        print(f"Image data ex: {self.image_data[0]}")

    def load_action_list_from_metadata(self):
        with open(self.classes_path, 'r') as f:
            data = json.load(f)

        actions = data['actions']
        return actions

    def load_image_data_in_chunks(self):
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for chunk_name in self.chunk_list:
                chunk_path = os.path.join(self.root_dir, chunk_name)
                if os.path.exists(chunk_path) and os.path.isdir(chunk_path):
                    futures.extend(
                        [
                            executor.submit(
                                self.collect_files,
                                chunk_path,
                                filename,
                                chunk_name,
                            )
                            for filename in os.listdir(chunk_path)
                            if filename.endswith(".jpg")
                        ]
                    )
            for future in tqdm(futures, total=len(futures), desc="Loading image paths"):
                result = future.result()
                if result:
                    self.image_data.append(result)

    def collect_files(self, chunk_path, filename, chunk_name):
        img_file = os.path.join(chunk_path, filename)
        txt_file = os.path.join(chunk_path, filename.replace('.jpg', '.txt'))

        with open(txt_file, 'r') as f:
            caption = f.read().strip()

        check = False

        for action in self.action_list:
            if action in caption:
                check = True
                break

        scenarioAction_annot = chunk_name

        if os.path.exists(img_file) and check:
            return img_file, scenarioAction_annot
        return None

    def load_image_data(self, img_annot_pair):
        img_file, annot_class = img_annot_pair
        image = Image.open(img_file)

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


class CC3MwLLaVAResp_Dataset(Dataset):
    def __init__(
        self,
        root_dir,
        start_chunk,
        end_chunk,
        storage_dir,
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

        self.chunk_list = [x for x in range(332)]
        self.chunk_list = [str(y).zfill(5) for y in self.chunk_list][
            start_chunk : end_chunk + 1
        ]

        self.image_data = []
        self.classes_path = (
            "/home/zeta/Workbenches/ActionGenome_DataSet/cc3m_dataset/classes.json"
        )
        self.action_list = self.load_action_list_from_metadata()

        self.load_image_data_in_chunks()

        print(f"Found {len(self.image_data)} Files.")
        print(f"Image data ex: {self.image_data[0]}")

    def load_action_list_from_metadata(self):
        with open(self.classes_path, "r") as f:
            data = json.load(f)

        actions = data["actions"]
        return actions

    def load_image_data_in_chunks(self):
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for chunk_name in self.chunk_list:
                chunk_path = os.path.join(self.root_dir, chunk_name)
                if os.path.exists(chunk_path) and os.path.isdir(chunk_path):
                    futures.extend(
                        [
                            executor.submit(
                                self.collect_files,
                                chunk_path,
                                filename,
                                chunk_name,
                            )
                            for filename in os.listdir(chunk_path)
                            if filename.endswith(".jpg")
                        ]
                    )
            for future in tqdm(futures, total=len(futures), desc="Loading image paths"):
                result = future.result()
                if result:
                    self.image_data.append(result)

    def collect_files(self, chunk_path, filename, chunk_name):
        img_file = os.path.join(chunk_path, filename)
        txt_file = os.path.join(chunk_path, filename.replace(".jpg", ".txt"))

        with open(txt_file, "r") as f:
            caption = f.read().strip()

        check = False

        for action in self.action_list:
            if action in caption:
                check = True
                break

        gemmaResp_file = os.path.join(
            self.storage_dir,
            "gemma_jsons",
            chunk_name,
            filename.replace(".jpg", "_gemma.json"),
        )

        if os.path.exists(img_file) and check and os.path.exists(gemmaResp_file):
            try:
                with Image.open(img_file) as img:
                    img.verify()

                if os.path.getsize(gemmaResp_file) > 0:
                    return img_file, chunk_name, gemmaResp_file

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


class CC3MwDinoResp_Dataset(Dataset):
    def __init__(
        self,
        root_dir,
        start_chunk,
        end_chunk,
        storage_dir,
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

        self.chunk_list = [x for x in range(332)]
        self.chunk_list = [str(y).zfill(5) for y in self.chunk_list][
            start_chunk : end_chunk + 1
        ]
        self.classes_path = (
            "/home/zeta/Workbenches/ActionGenome_DataSet/cc3m_dataset/classes.json"
        )

        self.action_list = self.load_action_list_from_metadata()

        self.image_data = []

        self.load_image_data_in_chunks()

        print(f"Found {len(self.image_data)} Files.")
        print(f"Image data ex: {self.image_data[0]}")

    def load_action_list_from_metadata(self):
        with open(self.classes_path, "r") as f:
            data = json.load(f)

        actions = data["actions"]
        return actions

    def load_image_data_in_chunks(self):
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for chunk_name in self.chunk_list:
                chunk_path = os.path.join(self.root_dir, chunk_name)
                if os.path.exists(chunk_path) and os.path.isdir(chunk_path):
                    futures.extend(
                        [
                            executor.submit(
                                self.collect_files,
                                chunk_path,
                                filename,
                                chunk_name,
                            )
                            for filename in os.listdir(chunk_path)
                            if filename.endswith(".jpg")
                        ]
                    )
            for future in tqdm(futures, total=len(futures), desc="Loading image paths"):
                result = future.result()
                if result:
                    self.image_data.append(result)

    def collect_files(self, chunk_path, filename, chunk_name):
        img_file = os.path.join(chunk_path, filename)
        txt_file = os.path.join(chunk_path, filename.replace(".jpg", ".txt"))

        with open(txt_file, "r") as f:
            caption = f.read().strip()

        check = False

        for action in self.action_list:
            if action in caption:
                check = True
                break

        gemmaResp_file = os.path.join(
            self.storage_dir,
            "gemma_jsons",
            chunk_name,
            filename.replace(".jpg", "_gemma.json"),
        )
        dinoResp_file = os.path.join(
            self.storage_dir,
            "dino_results",
            chunk_name,
            filename.replace(".jpg", "_grounding_dino.json"),
        )

        if (
            os.path.exists(img_file) and check
            and os.path.exists(gemmaResp_file)
            and os.path.exists(dinoResp_file)
        ):
            try:
                with Image.open(img_file) as img:
                    img.verify()

                if (
                    os.path.getsize(gemmaResp_file) > 0
                    and os.path.getsize(dinoResp_file) > 0
                ):
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
            prev_annotations = json.load(jfile)

        with open(dinoResp_file, "r") as jfile:
            dino_annotations = json.load(jfile)

        return img_file, image, prev_annotations, dino_annotations

    def __len__(self) -> int:
        """Return the total number of image files."""
        return len(self.image_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        """Generates one sample of data."""
        img_file, image, gemma_annotations, dino_annotations = self.load_image_data(
            self.image_data[idx]
        )
        return img_file, image, gemma_annotations, dino_annotations
