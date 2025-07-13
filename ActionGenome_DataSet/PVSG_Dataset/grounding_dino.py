import re
import os
import sys

sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-2]))

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import json
import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders.vidor_dataset import custom_collate_vidorwLLaVa, VidOrwLLaVAResp_Dataset
from dataloaders.epic_kitchen_dataset import EpicwLLaVAResp_Dataset, custom_collate_epicwLLaVa

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

torch.cuda.empty_cache()
# ---------------------------------------------------------------------------- #
#                                 Grounding DINO                               #
# ---------------------------------------------------------------------------- #


MODEL_NAME_DICT = {
    "dino_tiny": "IDEA-Research/grounding-dino-tiny",
    "dino_base": "IDEA-Research/grounding-dino-base",
}


class ObjDet_GroundingDINO:
    def __init__(self, name):
        self.className = str(name)
        return

    def create_batch_annotations(self, llava_responses):
        """
        For batch processing, text queries need to be lowercased + end with a dot
        """
        batch_annotations = [
            ". ".join(response["response"]["objects"]).lower()
            for response in llava_responses
        ]

        return batch_annotations

    def create_dataset(self, args):
        if "vidor" in args.root_dir.lower():
            image_dataset = VidOrwLLaVAResp_Dataset(
                root_dir=args.root_dir,
                storage_dir=args.storage_dir,
            )

            image_loader = DataLoader(
                image_dataset,
                batch_size=args.batch_size,
                num_workers=args.max_workers,
                shuffle=False,
                collate_fn=custom_collate_vidorwLLaVa,
            )
            
        elif "epic" in args.root_dir.lower():
            image_dataset = EpicwLLaVAResp_Dataset(
                root_dir=args.root_dir,
                storage_dir=args.storage_dir,
                start_chunk=args.start_chunk,
                end_chunk=args.end_chunk,
            )
            
            image_loader = DataLoader(
                image_dataset,
                batch_size=args.batch_size,
                num_workers=args.max_workers,
                shuffle=False,
                collate_fn=custom_collate_epicwLLaVa,
            )

        else:
            raise NotImplementedError

        return image_loader

    def nms(self, boxes, scores, iou_threshold):
        """
        Perform Non-Maximum Suppression (NMS) on the bounding boxes.

        Args:
            boxes (tensor): Tensor of shape (N, 4) containing the bounding boxes.
            scores (tensor): Tensor of shape (N,) containing the scores for each box.
            iou_threshold (float): IoU threshold for suppression.

        Returns:
            keep (tensor): Indices of the boxes to keep.
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        _, order = scores.sort(0, descending=True)

        keep = []
        while order.numel() > 0:
            if len(order.size()) == 0:
                i = order.item()
            else:
                i = order[0].item()
            keep.append(i)

            if order.numel() == 1:
                break

            xx1 = torch.max(x1[i], x1[order[1:]])
            yy1 = torch.max(y1[i], y1[order[1:]])
            xx2 = torch.min(x2[i], x2[order[1:]])
            yy2 = torch.min(y2[i], y2[order[1:]])

            w = torch.clamp(xx2 - xx1 + 1, min=0.0)
            h = torch.clamp(yy2 - yy1 + 1, min=0.0)

            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = (iou <= iou_threshold).nonzero().squeeze()
            if inds.numel() == 0:
                break

            order = order[inds + 1]

        return torch.tensor(keep, device=boxes.device)

    def perfom_nms_batch(self, batch_results, iou_threshold=0.5):
        new_results = []
        for data in batch_results:
            keep_indices = self.nms(data["boxes"], data["scores"], iou_threshold)
            if len(keep_indices) == 0:
                new_results.append(
                    {
                        "scores": data["scores"].tolist(),
                        "labels": data["labels"],
                        "boxes": data["boxes"].tolist(),
                    }
                )
                continue
            filtered_scores = data["scores"][keep_indices].tolist()
            filtered_labels = [data["labels"][i] for i in keep_indices]
            filtered_boxes = data["boxes"][keep_indices].tolist()
            new_results.append(
                {
                    "scores": filtered_scores,
                    "labels": filtered_labels,
                    "boxes": filtered_boxes,
                }
            )

        return new_results

    def detect_objects(self, args, image_loader=None):
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        if image_loader is None:
            image_loader = self.create_dataset(args)

        processor = AutoProcessor.from_pretrained(
            MODEL_NAME_DICT[args.model_name], cache_dir=args.cache_dir
        )
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            MODEL_NAME_DICT[args.model_name], cache_dir=args.cache_dir
        ).to(device)
        model.eval()

        with open(args.color_mapping_json, "r") as f:
            object_colors = json.load(f)

        for key in object_colors:
            object_colors[key] = tuple(object_colors[key])

        with tqdm(image_loader) as pbar:
            for img_files, batch_images, _, llava_annotations in pbar:
                annotations = llava_annotations[0]
                if not annotations["response"]["objects"]:
                    continue

                if not annotations["response"]["relations"]:
                    print(f"Skipping file due to empty relations: {img_files}")
                    continue

                batch = len(batch_images)
                try:
                    batch_annotations = self.create_batch_annotations(llava_annotations)
                except:
                    print(
                        "Error in batch_annotations. Skipping results for this batch."
                    )
                    continue

                inputs = processor(
                    images=batch_images,
                    text=batch_annotations,
                    return_tensors="pt",
                    padding=True,
                ).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)

                original_sizes = []
                for img in batch_images:
                    _, h, w = img.shape
                    original_sizes.append([h, w])

                results = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=0.25,
                    text_threshold=0.25,
                    target_sizes=original_sizes,
                )

                new_results = self.perfom_nms_batch(results, iou_threshold=0.4)

                def tensor_to_pil_image(tensor):
                    image_np = tensor.numpy()
                    image_np = image_np.astype(np.uint8)
                    image_pil = Image.fromarray(np.transpose(image_np, (1, 2, 0)))
                    return image_pil

                if args.img_save:
                    for b in range(len(batch_images)):
                        img = tensor_to_pil_image(255 - batch_images[b])
                        draw = ImageDraw.Draw(img)

                        try:
                            font = ImageFont.truetype("arial.ttf", 100)
                        except IOError:
                            font = ImageFont.load_default()

                        item = new_results[b]
                        boxes = item["boxes"]
                        labels = item["labels"]
                        scores = list(item["scores"])

                        bbox_color_default = "yellow"
                        txt_bg_color = "black"
                        text_fill = "white"

                        if boxes != []:
                            for box, label, score in zip(boxes, labels, scores):
                                bbox_color = object_colors.get(
                                    label, bbox_color_default
                                )
                                draw.rectangle(list(box), outline=bbox_color, width=8)
                                text = f"{label}: {score:.2f}"
                                text_bbox = draw.textbbox((0, 0), text, font=font)
                                text_width = text_bbox[2] - text_bbox[0]
                                text_height = text_bbox[3] - text_bbox[1]

                                padding = 10
                                draw.rectangle(
                                    [
                                        box[0],
                                        box[1] - text_height - padding,
                                        box[0] + text_width + padding,
                                        box[1],
                                    ],
                                    fill=txt_bg_color,
                                )
                                draw.text(
                                    (
                                        box[0] + padding // 2,
                                        box[1] - text_height - padding // 2,
                                    ),
                                    text,
                                    fill=text_fill,
                                    font=font,
                                )

                        os.makedirs(
                            os.path.join(args.storage_dir, "bbox_visualised"),
                            exist_ok=True,
                        )
                        impath = os.path.join(
                            args.storage_dir,
                            "bbox_visualised",
                            img_files[b].split("/")[-2],
                            os.path.basename(img_files[b]).replace(".png", "_bbox.png"),
                        )
                        img.save(impath)

                for i in range(batch):
                    result_dict = new_results[i]
                    result_dict["img_file"] = img_files[i]

                    os.makedirs(
                        os.path.join(args.storage_dir, "dino_results"),
                        exist_ok=True,
                    )

                    if "vidor" in args.root_dir.lower():
                        os.makedirs(
                            os.path.join(
                                args.storage_dir,
                                "dino_results",
                                img_files[i].split("/")[-2],
                            ),
                            exist_ok=True,
                        )
                        json_path = os.path.join(
                            args.storage_dir,
                            "dino_results",
                            img_files[i].split("/")[-2],
                            os.path.basename(img_files[i]).replace(
                                ".png", "_grounding_dino.json"
                            ),
                        )
                        
                    elif "epic" in args.root_dir.lower():
                        os.makedirs(
                                os.path.join(
                                args.storage_dir,
                                "dino_results",
                                img_files[i].split("/")[-2],
                            ), exist_ok=True
                        )
                        json_path = os.path.join(
                            args.storage_dir, 
                            "dino_results",
                            img_files[i].split("/")[-2],
                            os.path.basename(img_files[i]).replace(
                                ".png", "_grounding_dino.json"
                            ),
                        )
                        
                    else:
                        raise NotImplementedError

                    print(f"Json stored at: {json_path}")
                    self.save_results_to_json(result_dict, json_path)

    def save_results_to_json(self, results, output_file):
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="dino_base",
        help="Model name from the list of available models",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/scratch/saali/Datasets/OpenPVSG/data/epic_kitchen",
    )
    parser.add_argument(
        "--storage_dir",
        type=str,
        default="/scratch/saali/Datasets/pvsg_results/epic_kitchen",
    )
    parser.add_argument(
        "--color_mapping_json",
        type=str,
        default="/scratch/saali/Datasets/coco/color_mapping.json",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--cache_dir",
        default="/scratch/saali/Datasets/.cache",
    )
    parser.add_argument(
        '--start_chunk',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--end_chunk',
        type=int,
        default=54,
    )
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument(
        "--img_save", type=bool, default=False, help="Save images with bounding boxes"
    )
    args = parser.parse_args()

    objDet_GroundingDINO = ObjDet_GroundingDINO("ObjDet_GroundingDINO")
    objDet_GroundingDINO.detect_objects(args)


if __name__ == "__main__":
    main()
