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


def custom_collate_epicwLLaVa(batch):
    img_files, images, annotations, llava_annots = zip(*batch)
    return list(img_files), list(images), list(annotations), list(llava_annots)

def custom_collate_epicwDino(batch):
    img_files, images, annotations, dino_annots = zip(*batch)
    return list(img_files), list(images), list(annotations), list(dino_annots)

def custom_collate_epicForActionCaption(batch):
    img_files, images, gemma_annots = zip(*batch)
    return list(img_files), list(images), list(gemma_annots)

class EpicKitchenDataset(Dataset):
    def __init__(self, root_dir, storage_dir, pvsg_json, need_frames, need_jsons, num_workers=4):
        self.root_dir = root_dir
        self.storage_dir = storage_dir
        self.num_workers = num_workers

        with open(pvsg_json, 'r') as f:
            self.pvsg = json.load(f)

        self.video_ids = self.pvsg['split']['epic_kitchen']['train'] + self.pvsg['split']['epic_kitchen']['val']

        self.data = {
            data_dict['video_id']: data_dict for data_dict in self.pvsg['data'] if data_dict['video_id'] in self.video_ids
        }

        self.video_folder = os.path.join(self.root_dir, 'videos')

        if need_frames:
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = [
                    executor.submit(self.get_frames, vid, self.data, self.video_folder)
                    for vid in self.video_ids
                ]

                for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting frames"):
                    future.result()

        if need_jsons:
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = {
                    executor.submit(self.process_and_save_video, video_id): video_id
                    for video_id in self.video_ids
                }

                for future in tqdm(as_completed(futures), total=len(self.video_ids), desc="Processing videos"):
                    video_id = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error processing video {video_id}: {e}")

    def get_frames(self, vid, video_data, video_folder):
        video_dict = video_data[vid]
        video_path = os.path.join(video_folder, f"{vid}.MP4")
        os.makedirs(os.path.join(video_folder, vid), exist_ok=True)
        output_folder = os.path.join(video_folder, vid)

        fps = video_dict['meta']['fps']

        command = [
            'ffmpeg', '-i', video_path, '-vf', f'fps={fps}', f'{output_folder}/%04d.png'
        ]

        try:
            subprocess.run(command, check=True)
            print(f"Frames extracted for vid: {vid}")

        except Exception as e:
            print(f"Skipping extracting frames for vid: {vid} due to exception: {e}")

    def process_video(self, video_id):
        num_frames = self.data[video_id]["meta"]["num_frames"]
        object_mappings = {
            obj_dict["object_id"]: obj_dict["category"]
            for obj_dict in self.data[video_id]["objects"]
        }
        frame_wise_relations = {x: [] for x in range(num_frames)}

        for relation in self.data[video_id]["relations"]:
            time_ranges = relation[3]

            for time_range in time_ranges:
                low = time_range[0]
                high = time_range[1]
                for x in range(low, high + 1):
                    if x in frame_wise_relations.keys():
                        frame_wise_relations[x].append(
                            [
                                f"{object_mappings[relation[0]]}",
                                f"{relation[2]}",
                                f"{object_mappings[relation[1]]}",
                            ]
                        )

        valid_frames = [
            frame_num
            for frame_num in frame_wise_relations.keys()
            if frame_wise_relations[frame_num]
        ]

        return frame_wise_relations, valid_frames

    def process_and_save_video(self, video_id):
        frame_wise_relations, valid_frames = self.process_video(video_id)

        for valid_frame in valid_frames:
            img_path = os.path.join(
                self.video_folder, video_id, f"{str(valid_frame).zfill(4)}.png"
            )

            output_path = os.path.join(
                self.storage_dir,
                "gemma_jsons",
                f"{video_id}",
                f"{str(valid_frame).zfill(4)}_gemma.json",
            )

            if os.path.exists(output_path):
                print(f'File already exists: {output_path}')
                continue

            objects = set()
            for rel in frame_wise_relations[valid_frame]:
                objects.add(rel[0])
                objects.add(rel[2])
            output_dic = {
                "image": img_path,
                "response": {
                    "action": "",
                    "objects": list(objects),
                    "relations": frame_wise_relations[valid_frame],
                },
            }
            os.makedirs(os.path.join(self.storage_dir, "gemma_jsons"), exist_ok=True)
            os.makedirs(os.path.join(self.storage_dir, "gemma_jsons", video_id), exist_ok=True)
            with open(
                output_path,
                "w",
            ) as f:
                json.dump(output_dic, f, indent=4)


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

class EpicwLLaVAResp_Dataset(Dataset):
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

        self.video_folder = os.path.join(self.root_dir, 'videos')

        self.all_folders = [
            f for f in os.listdir(self.video_folder) if os.path.isdir(os.path.join(self.video_folder, f))
        ]
        self.all_folders = sorted(self.all_folders)
        
        self.chunk_list = self.all_folders[self.start_chunk: min(len(self.all_folders) - 1, self.end_chunk + 1)]

        self.image_data = []

        self.load_image_data_in_chunks()

        print(f"Found {len(self.image_data)} files.")
        print(f"Image Data Ex: {self.image_data[0]}")

    def load_image_data_in_chunks(self):
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []

            for folder in self.chunk_list:
                folder_path = os.path.join(self.video_folder, folder)
                if os.path.exists(folder_path) and folder in self.chunk_list:
                    futures.extend(
                        [
                            executor.submit(
                                self.collect_files, folder, folder_path, filename
                            )
                            for filename in os.listdir(folder_path)
                            if filename.endswith(".jpg") or filename.endswith(".png")
                        ]
                    )

            for future in tqdm(futures, total=len(futures), desc="Loading image paths"):
                result = future.result()
                if result:
                    self.image_data.append(result)

    def collect_files(self, folder, folder_path, filename):
        img_file = os.path.join(folder_path, filename)
        gemmaResp_file = os.path.join(
            self.storage_dir,
            "gemma_jsons",
            folder,
            filename.replace(".png", "_gemma.json"),
        )

        if os.path.exists(img_file) and os.path.exists(gemmaResp_file):
            try:
                with Image.open(img_file) as img:
                    img.verify()

                if os.path.getsize(gemmaResp_file) > 0:
                    return img_file, "", gemmaResp_file

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

class EpicwDinoResp_Dataset(Dataset):
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

        self.video_folder = os.path.join(self.root_dir, 'videos')

        self.all_folders = [
            f for f in os.listdir(self.video_folder) if os.path.isdir(os.path.join(self.video_folder, f))
        ]
        self.all_folders = sorted(self.all_folders)

        self.chunk_list = self.all_folders[self.start_chunk: min(len(self.all_folders) - 1, self.end_chunk + 1)]

        self.image_data = []

        self.load_image_data_in_chunks()

        print(f"Found {len(self.image_data)} files.")
        print(f"Image Data Ex: {self.image_data[0]}")

    def load_image_data_in_chunks(self):
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []

            for folder in self.chunk_list:
                folder_path = os.path.join(self.video_folder, folder)
                if os.path.exists(folder_path) and folder in self.chunk_list:
                    futures.extend(
                        [
                            executor.submit(
                                self.collect_files, folder, folder_path, filename
                            )
                            for filename in os.listdir(folder_path)
                            if filename.endswith(".jpg") or filename.endswith(".png")
                        ]
                    )

            for future in tqdm(futures, total=len(futures), desc="Loading image paths"):
                result = future.result()
                if result:
                    self.image_data.append(result)

    def collect_files(self, folder, folder_path, filename):
        img_file = os.path.join(folder_path, filename)
        gemmaResp_file = os.path.join(
            self.storage_dir,
            "gemma_jsons",
            folder,
            filename.replace(".png", "_gemma.json"),
        )
        dinoResp_file = os.path.join(
            self.storage_dir,
            "dino_results",
            folder,
            filename.replace(".png", "_grounding_dino.json"),
        )

        if os.path.exists(img_file) and os.path.exists(gemmaResp_file) and os.path.exists(dinoResp_file):
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


class EpicForActionCaption_Dataset(Dataset):
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

        self.video_folder = os.path.join(self.root_dir, "videos")

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
                if os.path.exists(folder_path) and folder in self.chunk_list:
                    futures.extend(
                        [
                            executor.submit(
                                self.collect_files, folder, folder_path, filename
                            )
                            for filename in os.listdir(folder_path)
                            if filename.endswith(".jpg") or filename.endswith(".png")
                        ]
                    )

            for future in tqdm(futures, total=len(futures), desc="Loading image paths"):
                result = future.result()
                if result:
                    self.image_data.append(result)

    def collect_files(self, folder, folder_path, filename):
        img_file = os.path.join(folder_path, filename)
        gemmaResp_file = os.path.join(
            self.storage_dir,
            "gemma_jsons",
            folder,
            filename.replace(".png", "_gemma.json"),
        )

        if os.path.exists(img_file) and os.path.exists(gemmaResp_file):
            try:
                with Image.open(img_file) as img:
                    img.verify()

                if os.path.getsize(gemmaResp_file) > 0:
                    return img_file, gemmaResp_file

            except Exception as e:
                pass
        return None

    def load_image_data(self, img_annot_pair):
        img_file, gemmaResp_file = img_annot_pair
        image = Image.open(img_file)

        return img_file, image, gemmaResp_file

    def __len__(self) -> int:
        """Return the total number of image files."""
        return len(self.image_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        """Generates one sample of data."""
        img_file, image, gemma_path = self.load_image_data(
            self.image_data[idx]
        )
        return img_file, image, gemma_path
