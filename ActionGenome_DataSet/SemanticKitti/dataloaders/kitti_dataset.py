import os
import json
from PIL import Image, ImageOps
from torchvision import transforms
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def custom_collate_kitti(batch):
    img_files, images, annotations = zip(*batch)
    return list(img_files), list(images), list(annotations)


def custom_collate_kittiwLLaVa(batch):
    img_files, images, annotations, llava_annots = zip(*batch)
    return list(img_files), list(images), list(annotations), list(llava_annots)


def custom_collate_kittiwDino(batch):
    img_files, images, annotations, dino_annots = zip(*batch)
    return list(img_files), list(images), list(annotations), list(dino_annots)

def custom_collate_kittiwPosNeg(batch):
    gemma_files = batch
    return list(gemma_files)


def resize_with_zero_padding(image, new_size):
    return ImageOps.pad(
        image, new_size, method=Image.NEAREST, color=0, centering=(0.5, 0.5)
    )


def is_json_file_empty(file_path):
    """Check if the given JSON file is empty."""
    try:
        # Check if file exists and is not empty
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            with open(file_path, "r") as f:
                first_char = f.read(1)  # Read the first character
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


class kitti_Dataset(Dataset):
    def __init__(
        self,
        root_dir,
        start_chunk,
        end_chunk,
        which_chunk,
        image_shape=(512, 512),
        chunk_json_path="",
        metadata_json_path="",
        storage_dir="",
        split="train",
        transform=None,
    ):
        """
        Args:
        root_dir (string): Directory with all the images.
        image_shape (tuple): Desired size of the images.
        chunk_json_path (string): Path to the JSON file containing chunked filenames.
        metadata_json_path (string): Path to the JSON file containing image metadata.
        transform (callable, optional): Optional transform to be applied on a sample.
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
        self.chunk_json_path = chunk_json_path
        self.metadata_json_path = metadata_json_path
        self.start_chunk = start_chunk
        self.end_chunk = end_chunk
        self.split = split
        self.which_chunk = which_chunk

        self.chunk_list = ['image_2', 'image_3']
        self.chunk_list = [self.chunk_list[which_chunk]]
        print(f"Chunk List: {self.chunk_list}")

        self.image_data = []
        self.load_image_data_in_chunks()

        print(f"Found {len(self.image_data)} Files.")
        print(f"Image data ex: {self.image_data[0]}")

    def load_image_data_in_chunks(self):
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for chunk_name in self.chunk_list:
                filenames = os.listdir(os.path.join(self.root_dir, chunk_name))
                futures.extend(
                    executor.submit(self.process_chunk, chunk_name, filename)
                    for filename in filenames
                )

            for future in tqdm(futures, total=len(futures), desc="Loading image data"):
                result = future.result()
                if result:
                    self.image_data.append(result)

    def process_chunk(self, chunk_name, filename):
        img_path = os.path.join(self.root_dir, chunk_name, filename)
        scenario_actionAnnot = chunk_name
        if os.path.exists(img_path):
            try:
                with Image.open(img_path) as img:
                    img.verify()
                return (img_path, scenario_actionAnnot)
            except Exception as e:
                pass

        return None

    def load_image_data(self, img_annot_pair):
        img_file, annot_class = img_annot_pair
        image = Image.open(img_file)

        return img_file, image, annot_class

    def __len__(self) -> int:
        """Return the total number of image files."""
        return len(self.image_data)

    def __getitem__(self, idx: int):
        """Generates one sample of data."""
        img_file, image, scenarioAction_class = self.load_image_data(
            self.image_data[idx]
        )
        return img_file, image, scenarioAction_class


class kittiwLLaVAResp_Dataset(Dataset):
    def __init__(
        self,
        root_dir,
        chunk_json_path,
        metadata_json_path,
        start_chunk,
        end_chunk,
        storage_dir="",
        split="train",
        image_shape=None,
        transform=None,
    ):
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
        self.transform = transform or transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.storage_dir = storage_dir
        self.chunk_json_path = chunk_json_path
        self.split = split

        self.metadata_json_path = metadata_json_path
        self.chunk_list = ["image_2", "image_3"]
        self.image_data = []

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for chunk_name in self.chunk_list:
                filenames = os.listdir(os.path.join(self.root_dir, chunk_name))
                futures.extend(
                    [
                        executor.submit(self.collect_files, chunk_name, filename)
                        for filename in filenames
                    ]
                )

            for future in tqdm(
                futures, total=len(futures), desc="Loading image paths"
            ):
                result = future.result()
                if result:
                    self.image_data.append(result)

        print(f"Found {len(self.image_data)} Files.")
        print(f"Image data ex: {self.image_data[0]}")

    def collect_files(self, chunk_name, filename):
        img_path = os.path.join(self.root_dir, chunk_name, filename)
        scenarioAction_annot = chunk_name
        llamaResp_file = os.path.join(self.storage_dir, "gemma_jsons", chunk_name, filename.replace(".png", "_gemma.json"))

        if os.path.exists(img_path) and os.path.exists(llamaResp_file):
            try:
                with Image.open(img_path) as img:
                    img.verify()

                if os.path.getsize(llamaResp_file) > 0:
                    return img_path, scenarioAction_annot, llamaResp_file
                else:
                    print(f"Skipping empty or invalid gemma response file: {llamaResp_file}")

            except Exception as e:
                # print(f"Skipping corrupted or unreadable image: {img_file} due to exception: {e}")r
                pass

        return None

    def load_image_data(self, img_annot_pair):
        img_file, annot_class, llavaResp_file = img_annot_pair
        image = Image.open(img_file).convert("RGB")
        image = self.transform(image)
        if image.max() <= 1:
            image = (1 - image) * 255
            image = image.int()

        with open(llavaResp_file, "r") as jfile:
            llava_annotations = json.load(jfile)

        return img_file, image, annot_class, llava_annotations

    def __len__(self) -> int:
        """Return the total number of image files."""
        return len(self.image_data)

    def __getitem__(self, idx: int):
        """Generates one sample of data."""
        img_file, image, scenarioAction_class, llava_annotation = self.load_image_data(
            self.image_data[idx]
        )
        return img_file, image, scenarioAction_class, llava_annotation


class kittiwDinoResp_Dataset(Dataset):
    def __init__(
        self,
        root_dir,
        chunk_json_path,
        metadata_json_path,
        start_chunk,
        end_chunk,
        storage_dir="",
        image_shape=None,
        transform=None,
        split='train',
    ):
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
        self.transform = transform or transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.storage_dir = storage_dir
        self.chunk_json_path = chunk_json_path
        self.start_chunk = start_chunk
        self.end_chunk = end_chunk
        self.split = split

        self.chunk_list = ["image_2", "image_3"]

        self.metadata_json_path = metadata_json_path
        self.image_data = []

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for chunk_name in self.chunk_list:
                filenames = os.listdir(os.path.join(self.root_dir, chunk_name))
                futures.extend(
                    [
                        executor.submit(self.collect_files, chunk_name, filename)
                        for filename in filenames
                    ]
                )

            for future in tqdm(futures, total=len(futures), desc="Loading image paths"):
                result = future.result()
                if result:
                    self.image_data.append(result)

        print(f"Found {len(self.image_data)} Files.")
        print(f"Image data ex: {self.image_data[0]}")

    def collect_files(self, chunk_name, filename):
        img_path = os.path.join(self.root_dir, chunk_name, filename)
        scenarioAction_annot = chunk_name
        llamaResp_file = os.path.join(
            self.storage_dir,
            "gemma_jsons",
            chunk_name,
            filename.replace(".png", "_gemma.json"),
        )
        dinoResp_file = os.path.join(
            self.storage_dir,
            "dino_results",
            chunk_name,
            filename.replace(".png", "_grounding_dino.json"),
        )

        if (
            os.path.exists(img_path)
            and os.path.exists(llamaResp_file)
            and os.path.exists(dinoResp_file)
        ):
            try:
                with Image.open(img_path) as img:
                    img.verify()

                if os.path.getsize(llamaResp_file) > 0 and os.path.getsize(dinoResp_file) > 0:
                    return img_path, llamaResp_file, dinoResp_file

            except Exception as e:
                pass

        return None

    def load_image_data(self, img_json_triple):
        img_file, json_file, dinoResp_file = img_json_triple
        image = Image.open(img_file).convert("RGB")
        image = self.transform(image)
        if image.max() <= 1:
            image = (
                1 - image
            ) * 255
            image = image.int()

        with open(json_file, "r") as jfile:
            prev_annotations = json.load(jfile)

        with open(dinoResp_file, "r") as jfile:
            dino_annotations = json.load(jfile)

        return img_file, image, prev_annotations, dino_annotations

    def __len__(self) -> int:
        """Return the total number of image files."""
        return len(self.image_data)

    def __getitem__(self, idx):
        """Generates one sample of data."""
        img_file, image, prev_annotations, dino_annotations = self.load_image_data(
            self.image_data[idx]
        )
        return img_file, image, prev_annotations, dino_annotations

class kittiforPosNegRelations_Dataset(Dataset):
    def __init__(
        self,
        storage_dir="",
        split="train",
    ):
        """
        Args:
            root_dir (string): Directory with all the images and JSON files.
            start_chunk (int): Index of the Starting Chunk
            end_chunk (int): Index of the Ending Chunk
            image_shape (tuple): Desired size of the images
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.storage_dir = storage_dir
        self.split = split

        self.image_data = []
        
        self.collect_files()
        
        print(f"Found {len(self.image_data)} Files.")

    def collect_files(self):
        for file in os.listdir(os.path.join(self.storage_dir, "gemma_jsons")):
            self.image_data.append(os.path.join(self.storage_dir, "gemma_jsons", file))

    def load_image_data(self, gemma_file):
        gemmaResp_file = gemma_file
        return gemmaResp_file

    def __len__(self) -> int:
        """Return the total number of image files."""
        return len(self.image_data)

    def __getitem__(self, idx: int):
        """Generates one sample of data."""
        gemmaResp_file = self.load_image_data(
            self.image_data[idx]
        )
        return gemmaResp_file
