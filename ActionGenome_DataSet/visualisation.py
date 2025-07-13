from PIL import Image, ImageDraw, ImageFont
import sys
import matplotlib.pyplot as plt
import argparse
import os
import json
import torchvision.transforms as T
import numpy as np
from scipy.ndimage import gaussian_filter, center_of_mass

def tensor_to_pil_image(tensor):
    image_np = tensor.numpy()
    image_np = image_np.astype(np.uint8)
    image_pil = Image.fromarray(np.transpose(image_np, (1, 2, 0)))
    return image_pil

def get_image_tensor(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose(
        [
            T.ToTensor(),
        ]
    )
    image = transform(image)
    if image.max() <= 1:
        image = (1 - image) * 255
        image = image.int()
    return image


def visualise_bbox(image_path, dino_json_path, color_mapping_json, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    image = get_image_tensor(image_path)

    img = tensor_to_pil_image(255 - image)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 100)
    except IOError:
        font = ImageFont.load_default()

    with open(color_mapping_json, "r") as f:
        object_colors = json.load(f)

    with open(dino_json_path) as f:
        data = json.load(f)

    boxes = data["boxes"]
    labels = data["labels"]
    scores = list(data["scores"])

    bbox_color_default = [255, 255, 0]  # yellow in RGB
    txt_bg_color = "black"
    text_fill = "white"

    if boxes != []:
        for box, label, score in zip(boxes, labels, scores):
            bbox_color = object_colors.get(label, bbox_color_default)
            draw.rectangle(list(box), outline=tuple(bbox_color), width=8)
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
                (box[0] + padding // 2, box[1] - text_height - padding // 2),
                text,
                fill=text_fill,
                font=font,
            )

    os.makedirs(output_dir, exist_ok=True)
    impath = os.path.join(
        output_dir, os.path.basename(image_path).replace(".jpg", "_bbox.jpg")
    )
    img.save(impath)

    return

def show_mask(mask, ax, color):
        h, w = mask.shape[-2:]
        if len(mask.shape) == 3:
            mask = mask[0]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

def show_masks_on_image(image_path, npz_file, output_path, colorMappingPath):
    raw_image = get_image_tensor(image_path)
    masks = np.load(npz_file)["mask"]
    scores = np.load(npz_file)["score"]
    labels = np.load(npz_file)["labels"]
    with open(colorMappingPath, "r") as f:
        object_colors = json.load(f)
    if len(masks.shape) == 4:
        masks = masks.squeeze()
    if scores.shape[0] == 1:
        scores = scores.squeeze()

    raw_image = (255 - raw_image).permute(1, 2, 0)

    _, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(np.array(raw_image))

    for i, label in enumerate(labels):
        max_idx = np.argmax(scores[i]) if len(scores) > 1 else 0
        mask = masks[i][max_idx]

        color = (
            np.array(
                object_colors.get(
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

        show_mask(mask, ax, color)

    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()

def process_npz_file(npz_file):
    data = np.load(npz_file)
    masks = data["mask"]
    scores = data["score"]
    labels = data["labels"]
    
    if masks.shape[0] != scores.shape[0] or masks.shape[0] != len(labels):
        raise ValueError("Number of masks, scores and labels should be the same")

    selected_masks = []

    for i in range(len(labels)):
        max_idx = np.argmax(scores[i])
        selected_masks.append(masks[i][max_idx])

    return np.array(selected_masks), labels

def calculate_object_size(mask):
    return mask.sum()

def create_rbf_mask(mask, com, sigma=100):
    rbf_mask = np.zeros_like(mask, dtype=float)
    h, w = mask.shape

    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - com[1]) ** 2 + (y - com[0]) ** 2)

    rbf_mask = np.exp(-(distance**2) / (2 * sigma**2))
    rbf_mask *= mask.astype(float)

    return rbf_mask

def create_distance_map(shape, com):
    h, w = shape
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - com[1]) ** 2 + (y - com[0]) ** 2)
    return distance


def find_closest_pair(boxes, indices_1, indices_2):
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
    img_path, npz_file, dino_path, output_folder, llama_path
):
    os.makedirs(output_folder, exist_ok=True)
    raw_image = get_image_tensor(img_path)
    with open(llama_path, "r") as f:
        gemma_data = json.load(f)

    with open(dino_path, "r") as f:
        dino_data = json.load(f)

    boxes = dino_data.get("boxes", [])

    relations = gemma_data.get("response", {}).get("relations", [])

    selected_masks, labels = process_npz_file(npz_file)

    raw_image = (255 - raw_image).permute(1, 2, 0).cpu().numpy()

    counter = 0

    for relation in relations:
        if len(relation) == 3:
            o1 = relation[0]
            o2 = relation[2]

            indices_o1 = np.where(np.array(labels) == o1)[0]
            indices_o2 = np.where(np.array(labels) == o2)[0]

            if len(indices_o1) > 0 and len(indices_o2) > 0:
                closest_indices = find_closest_pair(boxes, indices_o1, indices_o2)

                if closest_indices:
                    o1_idx, o2_idx = closest_indices
                    object1_mask = selected_masks[o1_idx]
                    object2_mask = selected_masks[o2_idx]

                    com1 = center_of_mass(object1_mask)
                    com2 = center_of_mass(object2_mask)

                    rbf_mask1 = create_rbf_mask(object1_mask, com1)
                    rbf_mask2 = create_rbf_mask(object2_mask, com2)

                    distance_map1 = create_distance_map(object1_mask.shape, com1)
                    distance_map2 = create_distance_map(object2_mask.shape, com2)

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

                    combined_mask = np.logical_or(object1_mask, object2_mask).astype(
                        np.uint8
                    )
                    final_image = np.where(
                        combined_mask[..., None], rbf_image, blurred_image
                    )

                    relation_output_path = os.path.join(
                        output_folder, os.path.basename(os.path.basename(img_path).replace(".jpg", f"_relation_{counter}.png"))
                    )
                    counter += 1
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

    print(f"Focused Regions Generated for {img_path}")
    return


def generate_relations_with_subplots(
    img_path, npz_file, dino_path, output_folder, llama_path
):
    os.makedirs(output_folder, exist_ok=True)
    raw_image = get_image_tensor(img_path)
    with open(llama_path, "r") as f:
        gemma_data = json.load(f)

    with open(dino_path, "r") as f:
        dino_data = json.load(f)

    boxes = dino_data.get("boxes", [])
    relations = gemma_data.get("response", {}).get("relations", [])
    selected_masks, labels = process_npz_file(npz_file)
    raw_image = (255 - raw_image).permute(1, 2, 0).cpu().numpy()

    # Prepare subplots
    num_relations = len(relations)
    num_columns = 2 
    num_rows = (
        num_relations + num_columns - 1
    ) // num_columns  
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, num_rows * 5))
    axes = axes.ravel()

    counter = 0

    for relation in relations:
        if len(relation) == 3:
            o1 = relation[0]
            o2 = relation[2]

            indices_o1 = np.where(np.array(labels) == o1)[0]
            indices_o2 = np.where(np.array(labels) == o2)[0]

            if len(indices_o1) > 0 and len(indices_o2) > 0:
                closest_indices = find_closest_pair(boxes, indices_o1, indices_o2)

                if closest_indices:
                    o1_idx, o2_idx = closest_indices
                    object1_mask = selected_masks[o1_idx]
                    object2_mask = selected_masks[o2_idx]

                    com1 = center_of_mass(object1_mask)
                    com2 = center_of_mass(object2_mask)

                    rbf_mask1 = create_rbf_mask(object1_mask, com1)
                    rbf_mask2 = create_rbf_mask(object2_mask, com2)

                    distance_map1 = create_distance_map(object1_mask.shape, com1)
                    distance_map2 = create_distance_map(object2_mask.shape, com2)

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

                    combined_mask = np.logical_or(object1_mask, object2_mask).astype(
                        np.uint8
                    )
                    final_image = np.where(
                        combined_mask[..., None], rbf_image, blurred_image
                    )

                    axes[counter].imshow(final_image)
                    axes[counter].set_title(f"{' '.join(relation)}")
                    axes[counter].axis("off")
                    counter += 1

    for i in range(counter, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, os.path.basename(img_path).replace(".jpg", "_subplots.png")))

    print(f"Focused Regions for {img_path} plotted successfully.")


def main():
    parser = argparse.ArgumentParser(description="Visualisation of Bounding Box and Mask Results")
    subparsers = parser.add_subparsers()

    ### BBOX VISUALISATION
    parser_bbox = subparsers.add_parser(
        "visualise_bbox", help="Visualise Bounding Box Results"
    )
    parser_bbox.add_argument(
        "--image_path",
        type=str,
        help="Path to the image",
        default="/home/zeta/Workbenches/ActionGenome_ScenarioCLIP/ActionGenome_DataSet/src/vanilla/vanilla.jpg",
    )
    parser_bbox.add_argument(
        "--dino_json_path",
        type=str,
        help="Path to the DINO JSON file",
        default="/home/zeta/Workbenches/ActionGenome_ScenarioCLIP/ActionGenome_DataSet/src/vanilla/dino_results/vanilla_grounding_dino.json",
    )
    parser_bbox.add_argument(
        "--color_mapping_json",
        type=str,
        help="Path to the color mapping JSON file",
        default="/home/zeta/Workbenches/ActionGenome_ScenarioCLIP/ActionGenome_DataSet/color_mapping.json",
    )
    parser_bbox.add_argument(
        "--output_dir",
        type=str,
        help="Path to the output directory",
        default="/home/zeta/Workbenches/ActionGenome_ScenarioCLIP/ActionGenome_DataSet/src/vanilla",
    )
    parser_bbox.set_defaults(func=visualise_bbox)

    ### MASK VISUALISATION
    parser_mask = subparsers.add_parser(
        "show_masks_on_image", help="Visualise Mask Results"
    )
    parser_mask.add_argument(
        "--image_path",
        type=str,
        help="Path to the image",
        default="/home/zeta/Workbenches/ActionGenome_ScenarioCLIP/ActionGenome_DataSet/src/vanilla/vanilla.jpg",
    )
    parser_mask.add_argument(
        "--npz_file",
        type=str,
        help="Path to the NPZ file",
        default="/home/zeta/Workbenches/ActionGenome_ScenarioCLIP/ActionGenome_DataSet/src/vanilla/sam_results/vanilla_grounding_sam.npz",
    )
    parser_mask.add_argument(
        "--colorMappingPath",
        type=str,
        help="Path to the color mapping JSON file",
        default="/home/zeta/Workbenches/ActionGenome_ScenarioCLIP/ActionGenome_DataSet/color_mapping.json",
    )
    parser_mask.add_argument(
        "--output_path",
        type=str,
        help="Path to the output image",
        default="/home/zeta/Workbenches/ActionGenome_ScenarioCLIP/ActionGenome_DataSet/src/vanilla/vanilla_masked.jpg",
    )
    parser_mask.set_defaults(func=show_masks_on_image)

    ### FOCUSED REGION GENERATION
    parser_focused_regions = subparsers.add_parser(
        "generate_relations", help="Generate Focused Regions"
    )
    parser_focused_regions.add_argument(
        "--img_path",
        type=str,
        help="Path to the image",
        default="/home/zeta/Workbenches/ActionGenome_ScenarioCLIP/ActionGenome_DataSet/src/vanilla/vanilla.jpg",
    )
    parser_focused_regions.add_argument(
        "--npz_file",
        type=str,
        help="Path to the NPZ file",
        default="/home/zeta/Workbenches/ActionGenome_ScenarioCLIP/ActionGenome_DataSet/src/vanilla/sam_results/vanilla_grounding_sam.npz",
    )
    parser_focused_regions.add_argument(
        "--dino_path",
        type=str,
        help="Path to the DINO JSON file",
        default="/home/zeta/Workbenches/ActionGenome_ScenarioCLIP/ActionGenome_DataSet/src/vanilla/dino_results/vanilla_grounding_dino.json",
    )
    parser_focused_regions.add_argument(
        "--output_folder",
        type=str,
        help="Path to the output folder",
        default="./output",
    )
    parser_focused_regions.add_argument(
        "--llama_path",
        type=str,
        help="Path to the GEMMA JSON file",
        default="/home/zeta/Workbenches/ActionGenome_ScenarioCLIP/ActionGenome_DataSet/src/vanilla/gemma_jsons/vanilla_gemma.json",
    )
    parser_focused_regions.set_defaults(func=generate_relations)
    
    ### FOCUSED REGION GENERATION WITH SUBPLOTS
    parser_focused_regions_subplots = subparsers.add_parser(
        "generate_relations_with_subplots", help="Generate Focused Regions with Subplots"
    )
    parser_focused_regions_subplots.add_argument(
        "--img_path",
        type=str,
        help="Path to the image",
        default="/home/zeta/Workbenches/ActionGenome_ScenarioCLIP/ActionGenome_DataSet/src/vanilla/vanilla.jpg",
    )
    parser_focused_regions_subplots.add_argument(
        "--npz_file",
        type=str,
        help="Path to the NPZ file",
        default="/home/zeta/Workbenches/ActionGenome_ScenarioCLIP/ActionGenome_DataSet/src/vanilla/sam_results/vanilla_grounding_sam.npz",
    )
    parser_focused_regions_subplots.add_argument(
        "--dino_path",
        type=str,
        help="Path to the DINO JSON file",
        default="/home/zeta/Workbenches/ActionGenome_ScenarioCLIP/ActionGenome_DataSet/src/vanilla/dino_results/vanilla_grounding_dino.json",
    )
    parser_focused_regions_subplots.add_argument(
        "--output_folder",
        type=str,
        help="Path to the output folder",
        default="./output",
    )
    parser_focused_regions_subplots.add_argument(
        "--llama_path",
        type=str,
        help="Path to the GEMMA JSON file",
        default="/home/zeta/Workbenches/ActionGenome_ScenarioCLIP/ActionGenome_DataSet/src/vanilla/gemma_jsons/vanilla_gemma.json",
    )
    parser_focused_regions_subplots.set_defaults(func=generate_relations_with_subplots)

    args = parser.parse_args()

    if "func" not in args:
        parser.print_help()
        return

    func_args = {k: v for k, v in vars(args).items() if k != "func"}

    args.func(**func_args)


if __name__ == '__main__':
    main()
