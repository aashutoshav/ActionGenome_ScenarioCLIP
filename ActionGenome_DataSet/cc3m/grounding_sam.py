import re
import os
import sys
import argparse

sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-2]))

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

import re
import os
import json
import torch
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from dataloaders.cc3m_dataset import CC3MwDinoResp_Dataset, custom_collate_CC3MwDino
from transformers import SamModel, SamProcessor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, center_of_mass

# ---------------------------------------------------------------------------- #
#                                 Grounding SAM                                #
# ---------------------------------------------------------------------------- #


MODEL_NAME_DICT = {
    "sam_huge": "facebook/sam-vit-huge",  # largest model
    "sam_large": "facebook/sam-vit-large",  # medium model
    "sam_base": "facebook/sam-vit-base",  # base model (smallest)
}


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


class GroundedSam:
    def __init__(self, name):
        self.className = str(name)
        return

    def create_dataset(self, args):
        if "cc3m" in args.root_dir.lower():
            image_dataset = CC3MwDinoResp_Dataset(
                root_dir=args.root_dir,
                start_chunk=args.start_chunk,
                end_chunk=args.end_chunk,
                storage_dir=args.storage_dir,   
            )
            image_loader = DataLoader(
                image_dataset,
                batch_size=args.batch_size,
                num_workers=args.max_workers,
                shuffle=False,
                collate_fn=custom_collate_CC3MwDino,
            )

        else:
            raise NotImplementedError

        return image_loader

    def create_batch_input_boxes(self, input_boxes):
        batch_input_boxes = []
        box_mask = [len(boxes) for boxes in input_boxes]
        max_boxes = max(box_mask)
        for boxes in input_boxes:
            if len(boxes) < max_boxes:
                boxes += [[0, 0, 1, 1]] * (max_boxes - len(boxes))
            batch_input_boxes.append(torch.tensor(boxes))
        batch_input_boxes = torch.stack(batch_input_boxes)
        return batch_input_boxes, box_mask

    def generate_response(self, args, image_loader=None):

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        model = SamModel.from_pretrained(
            MODEL_NAME_DICT[args.model_name],
            cache_dir=args.cache_dir,
        ).to(device)
        model.eval()

        processor = SamProcessor.from_pretrained(
            MODEL_NAME_DICT[args.model_name],
            cache_dir=args.cache_dir,
        )

        if image_loader is None:
            image_loader = self.create_dataset(args)

        with tqdm(image_loader, desc="Grounded SAM") as pbar:
            for img_file, batch_images, _, dino_annotations in pbar:
                check = False
                if "coco" in args.root_dir.lower():
                    for file in img_file:
                        folder = os.path.basename(file).split(".")[0]
                        if not os.path.exists(os.path.join(args.storage_dir, "sam_results", folder)):
                            check = True
                            break
                        
                if "coco" in args.root_dir.lower() and not check:
                    print(f"Skipping Batch: {img_file} as already processed.")
                    continue
                
                if "multi" in args.root_dir.lower():
                    for file in img_file:
                        if not os.path.exists(
                            os.path.join(
                                args.storage_dir,
                                "sam_results",
                                file.split("/")[-3],
                                file.split("/")[-2],
                                os.path.basename(file).split(".")[0],
                            )
                        ):
                            check = True
                            break
                        
                if "multi" in args.root_dir.lower() and not check:
                    print(f"Skipping Batch: {img_file} as already processed.")
                    continue
                
                for file in img_file:
                    folder = os.path.basename(file).split(".")[0]
                    chunk = file.split("/")[-2]
                    if not os.path.exists(os.path.join(args.storage_dir, "sam_results", chunk, folder)):
                        check = True
                        break
                    
                if not check:
                    print(f"Skipping Batch: {img_file} as already processed.")
                    continue
                
                try:
                    input_boxes = [annot["boxes"] for annot in dino_annotations]
                    batch_input_boxes, box_mask = self.create_batch_input_boxes(
                        input_boxes
                    )

                    inputs = processor(
                        images=batch_images,
                        input_boxes=batch_input_boxes,
                        return_tensors="pt",
                    ).to("cuda")
                    with torch.no_grad():
                        outputs = model(**inputs)

                        masks = processor.image_processor.post_process_masks(
                            outputs.pred_masks.cpu(),
                            inputs["original_sizes"].cpu(),
                            inputs["reshaped_input_sizes"].cpu(),
                        )
                    scores = outputs.iou_scores

                    for b in range(len(batch_images)):
                        result_dict = {
                            "mask": masks[b].cpu().numpy(),
                            "score": scores[b].detach().cpu().numpy(),
                            "labels": dino_annotations[b]["labels"],
                            "box_mask": box_mask[b],
                        }

                        os.makedirs(
                            os.path.join(args.storage_dir, "sam_results"), exist_ok=True
                        )
                        os.makedirs(
                            os.path.join(
                                args.storage_dir, "sam_results", img_file[b].split("/")[-2]
                            ), exist_ok=True
                        )
                        npz_path = os.path.join(args.storage_dir, "sam_results", img_file[b].split("/")[-2], os.path.basename(img_file[b]).replace(".jpg", "_grounding_sam.npz"))
                        llama_path = os.path.join(args.storage_dir, "gemma_jsons", img_file[b].split("/")[-2], os.path.basename(img_file[b]).replace(".jpg", "_gemma.json"))
                        dino_path = os.path.join(args.storage_dir, "dino_results", img_file[b].split("/")[-2], os.path.basename(img_file[b]).replace(".jpg", "_grounding_dino.json"))

                        np.savez(npz_path, **result_dict)
                        
                        if is_json_file_empty(llama_path) or is_json_file_empty(
                            dino_path
                        ):
                            print(
                                f"Skipping Samples: {img_file} due to empty JSON file."
                            )
                            continue

                        self.generate_relations(
                            img_file[b],
                            batch_images[b],
                            npz_path,
                            dino_path,
                            npz_path.replace("_grounding_sam.npz", "_relations.png"),
                            llama_path,
                        )

                except Exception as e:
                    print(f"Skipping Samples: {img_file} due to exception: {e}")
                    continue

    def show_mask(self, mask, ax, color):
        # color = np.array([30 / 255, 144 / 255, 255 / 255, 0.8])
        h, w = mask.shape[-2:]
        if len(mask.shape) == 3:
            mask = mask[0]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_masks_on_image(self, raw_image, masks, scores, labels, output_path):
        if len(masks.shape) == 4:
            masks = masks.squeeze()
        if scores.shape[0] == 1:
            scores = scores.squeeze()

        raw_image = (255 - raw_image).permute(1, 2, 0)

        _, ax = plt.subplots(figsize=(15, 15))
        ax.imshow(np.array(raw_image))

        for i, label in enumerate(labels):
            max_idx = torch.argmax(scores[i]) if len(scores) > 1 else 0
            mask = masks[i][max_idx].cpu().detach().numpy()

            color = (
                np.array(
                    self.object_colors.get(
                        label, np.append(np.random.randint(0, 256, size=3), 0.8)
                    )
                )
                / 255.0
            )

            color = np.append(color, 0.8)
            if color.shape[0] == 3:
                color = np.append(color, 0.8)
            elif color.shape[0] > 4:
                color = color[:3]
                color = np.append(color, 0.6)
            elif color.shape[0] < 4:
                raise ValueError("Color must have 4 elements")

            self.show_mask(mask, ax, color)

        ax.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    def process_npz_file(self, npz_file):
        data = np.load(npz_file)
        masks = data["mask"]
        scores = data["score"]
        labels = data["labels"]

        selected_masks = []

        for i in range(len(labels)):
            max_idx = np.argmax(scores[i])
            selected_masks.append(masks[i][max_idx])

        return np.array(selected_masks), labels

    def calculate_object_size(self, mask):
        return mask.sum()

    def create_rbf_mask(self, mask, com, sigma=100):
        rbf_mask = np.zeros_like(mask, dtype=float)
        h, w = mask.shape

        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((x - com[1]) ** 2 + (y - com[0]) ** 2)

        rbf_mask = np.exp(-(distance**2) / (2 * sigma**2))
        rbf_mask *= mask.astype(float)

        return rbf_mask

    def create_distance_map(self, shape, com):
        h, w = shape
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((x - com[1]) ** 2 + (y - com[0]) ** 2)
        return distance

    def find_closest_pair(self, boxes, indices_1, indices_2):
        ans = float("inf")
        closest_pair = None

        for i in indices_1:
            for j in indices_2:
                box1_center = np.array(
                    [(boxes[i][0] + boxes[i][2]) / 2, (boxes[i][1] + boxes[i][3]) / 2]
                )
                box2_center = np.array(
                    [(boxes[j][0] + boxes[j][2]) / 2, (boxes[j][1] + boxes[j][3]) / 2]
                )

                distance = np.linalg.norm(box1_center - box2_center)

                if distance < ans:
                    ans = distance
                    closest_pair = (i, j)

        return closest_pair

    def generate_relations(
        self, img_path, raw_image, npz_file, dino_path, output_path, llama_path
    ):
        with open(llama_path, "r") as f:
            gemma_data = json.load(f)

        with open(dino_path, "r") as f:
            dino_data = json.load(f)

        boxes = dino_data.get("boxes", [])

        relations = gemma_data.get("response", {}).get("relations", [])

        selected_masks, labels = self.process_npz_file(npz_file)

        raw_image = (255 - raw_image).permute(1, 2, 0).cpu().numpy()

        counter = 0

        for relation in relations:
            if len(relation) == 3:
                o1 = relation[0]
                o2 = relation[2]

                indices_o1 = np.where(np.array(labels) == o1)[0]
                indices_o2 = np.where(np.array(labels) == o2)[0]

                if len(indices_o1) > 0 and len(indices_o2) > 0:
                    closest_indices = self.find_closest_pair(
                        boxes, indices_o1, indices_o2
                    )
                    
                    if closest_indices is None or len(closest_indices) != 2:
                        continue

                    if closest_indices:
                        o1_idx, o2_idx = closest_indices
                        
                        if o1_idx >= len(selected_masks) or o2_idx >= len(selected_masks):
                            continue
                        
                        object1_mask = selected_masks[o1_idx]
                        object2_mask = selected_masks[o2_idx]

                        com1 = center_of_mass(object1_mask)
                        com2 = center_of_mass(object2_mask)

                        rbf_mask1 = self.create_rbf_mask(object1_mask, com1)
                        rbf_mask2 = self.create_rbf_mask(object2_mask, com2)

                        distance_map1 = self.create_distance_map(
                            object1_mask.shape, com1
                        )
                        distance_map2 = self.create_distance_map(
                            object2_mask.shape, com2
                        )

                        total_dist = distance_map1 + distance_map2
                        weight1 = distance_map2 / total_dist
                        weight2 = distance_map1 / total_dist

                        blended_rbf_mask = rbf_mask1 * weight1 + rbf_mask2 * weight2
                        rbf_image = raw_image.copy()
                        for channel in range(3):
                            rbf_image[..., channel] = (
                                rbf_image[..., channel].astype(float) * blended_rbf_mask
                            ).astype(np.uint8)

                        blurred_image = gaussian_filter(raw_image, sigma=2)

                        combined_mask = np.logical_or(
                            object1_mask, object2_mask
                        ).astype(np.uint8)
                        final_image = np.where(
                            combined_mask[..., None], rbf_image, blurred_image
                        )

                        relation_output_path = output_path.replace(
                            ".png", f"_{counter}.png"
                        )
                        counter += 1
                        basename_folder = os.path.basename(img_path.split(".")[0])
                        base_path = os.path.join(
                            os.path.dirname(relation_output_path), basename_folder
                        )
                        os.makedirs(base_path, exist_ok=True)
                        relation_output_path = os.path.join(
                            base_path, os.path.basename(relation_output_path)
                        )
                        Image.fromarray(final_image.astype("uint8")).save(
                            relation_output_path
                        )

                        gemma_data.setdefault("focused_regions", {})
                        if relation_output_path not in gemma_data["focused_regions"]:
                            gemma_data["focused_regions"][relation_output_path] = {
                                "labels": [o1, o2],
                                "relation": relation,
                            }

                        with open(llama_path, "w") as json_file:
                            json.dump(gemma_data, json_file, indent=4)

                        print(f"Focused Regions Generated for {output_path}")
            else:
                print(f"Skipping relation: {relation}")

    def show_boxes_on_image(self, raw_image, boxes):
        plt.figure(figsize=(10, 10))
        plt.imshow(raw_image)
        for box in boxes:
            self.show_box(box, plt.gca())
        plt.axis("on")
        plt.show()

    def show_points_on_image(self, raw_image, input_points, input_labels=None):
        plt.figure(figsize=(10, 10))
        plt.imshow(raw_image)
        input_points = np.array(input_points)
        if input_labels is None:
            labels = np.ones_like(input_points[:, 0])
        else:
            labels = np.array(input_labels)
        self.show_points(input_points, labels, plt.gca())
        plt.axis("on")
        plt.show()

    def show_points_and_boxes_on_image(
        self, raw_image, boxes, input_points, input_labels=None
    ):
        plt.figure(figsize=(10, 10))
        plt.imshow(raw_image)
        input_points = np.array(input_points)
        if input_labels is None:
            labels = np.ones_like(input_points[:, 0])
        else:
            labels = np.array(input_labels)
        self.show_points(input_points, labels, plt.gca())
        for box in boxes:
            self.show_box(box, plt.gca())
        plt.axis("on")
        plt.show()

    def show_points_and_boxes_on_image(
        self, raw_image, boxes, input_points, input_labels=None
    ):
        plt.figure(figsize=(10, 10))
        plt.imshow(raw_image)
        input_points = np.array(input_points)
        if input_labels is None:
            labels = np.ones_like(input_points[:, 0])
        else:
            labels = np.array(input_labels)
        self.show_points(input_points, labels, plt.gca())
        for box in boxes:
            self.show_box(box, plt.gca())
        plt.axis("on")
        plt.show()

    def show_points(coords, labels, ax, marker_size=375):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(
            pos_points[:, 0],
            pos_points[:, 1],
            color="green",
            marker="*",
            s=marker_size,
            edgecolor="white",
            linewidth=1.25,
        )
        ax.scatter(
            neg_points[:, 0],
            neg_points[:, 1],
            color="red",
            marker="*",
            s=marker_size,
            edgecolor="white",
            linewidth=1.25,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="sam_huge",
        help="Model name from the list of available models",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/mnt/MIG_archive/Datasets/CC3M/cc3m",
        help="Root directory of the dataset",
    )
    parser.add_argument(
        "--storage_dir",
        type=str,
        default="/mnt/MIG_archive/Datasets/CC3M/cc3m/cc3m_results",
    )
    parser.add_argument("--start_chunk", type=int, default=0, help="Start chunk index")
    parser.add_argument("--end_chunk", type=int, default=1, help="End chunk index")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for processing"
    )
    parser.add_argument(
        "--cache_dir",
        default="/home/zeta/Workbenches/ActionGenome_ScenarioCLIP/ActionGenome_DataSet/assets/.cache",
        help="Cache directory for storing model checkpoints",
    )
    parser.add_argument(
        "--max_workers", type=int, default=8, help="Number of workers for dataloader"
    )

    args = parser.parse_args()
    gsam = GroundedSam("GroundedSam")
    gsam.generate_response(args)


if __name__ == "__main__":
    main()
