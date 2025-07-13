import numpy as np
import argparse
import torch
import matplotlib.pyplot as plt
import cv2
import os
import json
from PIL import Image, ImageOps
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from scipy.spatial.distance import cdist

def custom_collate_cocowSam(batch):
    img_files, gemma_files, npz_files = zip(*batch)
    return list(img_files), list(gemma_files), list(npz_files)

class COCOwSamResp_Dataset(Dataset):
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

        with open(self.chunk_json_path, "r") as f:
            self.chunk_data = json.load(f)

        self.metadata_json_path = metadata_json_path
        self.chunk_list = [key for key in self.chunk_data["train"].keys()][
            start_chunk : end_chunk + 1
        ]
        self.image_data = []

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for chunk in self.chunk_list:
                filenames = self.chunk_data["train"][chunk]
                futures.extend(
                    [
                        executor.submit(self.collect_files, chunk, filename)
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
        img_file = os.path.join(self.root_dir, "train2017", filename)
        gemma_file = os.path.join(
            self.storage_dir, "gemma_jsons", filename.replace(".jpg", "_gemma.json")
        )
        samResp_file = os.path.join(
            self.storage_dir, 
            "sam_results",
            filename.replace(".jpg", "_grounding_sam.npz"),
        )

        if (
            os.path.exists(img_file)
            and os.path.exists(samResp_file)
            and os.path.exists(gemma_file)
        ):
            try:
                with Image.open(img_file) as img:
                    img.verify()

                return img_file, gemma_file, samResp_file

            except Exception as e:
                pass

        return None

    def load_image_data(self, img_json_npz_triple):
        img_file, gemma_file, npz_file = img_json_npz_triple

        return img_file, gemma_file, npz_file

    def __len__(self) -> int:
        """Return the total number of image files."""
        return len(self.image_data)

    def __getitem__(self, idx):
        """Generates one sample of data."""
        img_file, gemma_file, npz_file = self.load_image_data(
            self.image_data[idx]
        )
        return img_file, gemma_file, npz_file

def create_dataset(args):
    image_dataset = COCOwSamResp_Dataset(
        root_dir=args.root_dir,
        chunk_json_path=args.chunk_json_path,
        metadata_json_path=args.metadata_json_path,
        storage_dir=args.storage_dir,
        start_chunk=args.start_chunk,
        end_chunk=args.end_chunk,
    )
    image_loader = DataLoader(
        image_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.max_workers,
        collate_fn=custom_collate_cocowSam,
    )

    return image_loader


def load_data(sam_path, img_path):
    """Load the SAM npz data and the original image."""
    npz_data = np.load(sam_path)
    original_image = cv2.imread(img_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    return npz_data, original_image


def create_output_dir(output_dir):
    """Create the output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)


def apply_combined_mask(original_image, combined_mask):
    """Apply a combined mask to the original image and return the overlay image."""
    colored_mask = np.zeros_like(original_image, dtype=np.uint8)
    colored_mask[combined_mask] = [0, 0, 255]  # Red mask for combined regions
    alpha = 0.8
    overlay_image = cv2.addWeighted(original_image, 1.0, colored_mask, alpha, 0)
    return overlay_image


def save_masked_image(overlay_image, selected_labels, output_dir):
    """Save the combined masked image to the output directory with a safe file name."""
    label_name = "_".join(selected_labels)
    safe_label_name = "".join(c if c.isalnum() else "_" for c in label_name)
    output_path = os.path.join(output_dir, f"relation_{safe_label_name}.png")
    plt.imsave(output_path, overlay_image)


def get_mask_centroid(mask):
    """Calculate the centroid of a mask."""
    y_indices, x_indices = np.where(mask)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return None  # No valid mask
    return np.array([np.mean(y_indices), np.mean(x_indices)])


def find_closest_pair(npz_data, selected_labels):
    """Find the closest pair of masks for the selected labels."""
    masks = npz_data["mask"]
    labels = npz_data["labels"]

    label1_masks = []
    label2_masks = []

    for i, label in enumerate(labels):
        if label == selected_labels[0]:
            label1_masks.append((masks[i][0], get_mask_centroid(masks[i][0])))
        elif label == selected_labels[1]:
            label2_masks.append((masks[i][0], get_mask_centroid(masks[i][0])))

    label1_masks = [m for m in label1_masks if m[1] is not None]
    label2_masks = [m for m in label2_masks if m[1] is not None]

    if not label1_masks or not label2_masks:
        print("No valid masks found for the selected labels.")
        return None, None

    label1_centroids = np.array([m[1] for m in label1_masks])
    label2_centroids = np.array([m[1] for m in label2_masks])

    distances = cdist(label1_centroids, label2_centroids)
    min_idx = np.unravel_index(np.argmin(distances), distances.shape)

    closest_mask1 = label1_masks[min_idx[0]][0]
    closest_mask2 = label2_masks[min_idx[1]][0]

    return closest_mask1, closest_mask2


def process_and_save_closest_pair(
    npz_data, original_image, output_dir, selected_labels
):
    """Process and save the mask for the closest pair of selected labels."""
    mask1, mask2 = find_closest_pair(npz_data, selected_labels)

    if mask1 is None or mask2 is None:
        print("Unable to find a valid closest pair.")
        return

    combined_mask = mask1 | mask2
    overlay_image = apply_combined_mask(original_image, combined_mask)
    save_masked_image(overlay_image, selected_labels, output_dir)

    print(f"Focused region between {selected_labels} saved at: '{output_dir}'")


def get_response(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    image_loader = create_dataset(args)

    with tqdm(image_loader, desc='Focused Regions Generation') as pbar:
        for img_files, gemma_files, npz_files in pbar:
            for b in range(len(img_files)):
                img_path = img_files[b]
                gemma_path = gemma_files[b]
                sam_path = npz_files[b]

                npz_data, original_image = load_data(sam_path, img_path)
                
                if npz_data['mask'].shape[0] != npz_data['labels'].shape[0]:
                    continue
                
                basename = os.path.basename(img_path).split(".")[0]
                output_dir = os.path.join(args.storage_dir, "sam_results", basename)
                create_output_dir(output_dir)
                
                with open(gemma_path, "r") as f:
                    gemma_data = json.load(f)
                    
                relations = gemma_data['response']['relations']
                
                for relation in relations:
                    if len(relation)==3:
                        label_1 = relation[0]
                        label_2 = relation[2]
                        selected_labels = [label_1, label_2]

                        process_and_save_closest_pair(npz_data, original_image, output_dir, selected_labels)
                print(f"Processed {img_path}")
                
    print(f"Focused Regions Generation Completed.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        # default="/home/zeta/Workbenches/ActionGenome_ScenarioCLIP/ActionGenome_DataSet/assets/GLAMM_dataset",
        # default="/home/zeta/Workbenches/ActionGenome_ScenarioCLIP/ActionGenome_DataSet/assets/MMT_dataset",
        # default="/home/zeta/Workbenches/ActionGenome_ScenarioCLIP/ActionGenome_DataSet/assets/coco/checktrain",
        default="/mnt/MIG_store/Datasets/coco",
        # default="/mnt/MIG_store/Datasets/epic-kitchen/e2",
        # default="/home/zeta/Workbenches/ActionGenome_ScenarioCLIP/ActionGenome_DataSet/assets/MMT_dataset",
        help="Root directory of the dataset",
    )
    parser.add_argument(
        "--storage_dir",
        type=str,
        # default="/mnt/MIG_store/Datasets/ActionGenome/results_29_9",
        default="./results",
        # default="./mmt_results",
        # default="./epic_results",
    )

    parser.add_argument(
        "--chunk_json_path",
        type=str,
        # default="/home/zeta/Workbenches/ActionGenome_ScenarioCLIP/ActionGenome_DataSet/dataset/train_chunks.json",
        default="./test_chunks.json",
    )
    parser.add_argument(
        "--metadata_json_path",
        type=str,
        default="/home/zeta/Workbenches/ActionGenome_ScenarioCLIP/ActionGenome_DataSet/dataset/train_details.json",
    )
    parser.add_argument("--start_chunk", type=int, default=0, help="Start chunk index")
    parser.add_argument("--end_chunk", type=int, default=0, help="End chunk index")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for processing"
    )
    parser.add_argument(
        "--max_workers", type=int, default=8, help="Number of workers for dataloader"
    )

    args = parser.parse_args()
    get_response(args)

if __name__ == '__main__':
    main()