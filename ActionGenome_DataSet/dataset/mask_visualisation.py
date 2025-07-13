'''
Visualising a mask on an image, given npz data and the original image.
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


def load_data(sam_path, img_path):
    """Load the SAM npz data and the original image."""
    npz_data = np.load(sam_path)
    original_image = cv2.imread(img_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    return npz_data, original_image


def create_output_dir(output_dir):
    """Create the output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)


def apply_mask(original_image, combined_mask):
    """Apply a mask to the original image and return the overlay image."""
    colored_mask = np.zeros_like(original_image, dtype=np.uint8)
    colored_mask[combined_mask] = [0, 0, 255]
    alpha = 0.8
    overlay_image = cv2.addWeighted(original_image, 1.0, colored_mask, alpha, 0)
    return overlay_image


def save_masked_image(overlay_image, selected_label, output_dir):
    """Save the masked image to the output directory with a safe file name."""
    safe_label_name = "".join(c if c.isalnum() else "_" for c in selected_label)
    output_path = os.path.join(output_dir, f"mask_{safe_label_name}.png")
    plt.imsave(output_path, overlay_image)


def find_label_masks(npz_data, selected_label):
    """Find and combine all masks for the selected label."""
    masks = npz_data["mask"]
    labels = npz_data["labels"]

    combined_mask = np.zeros(masks[0][0].shape, dtype=bool)

    for i, label in enumerate(labels):
        if label == selected_label:
            combined_mask |= masks[i][0]

    return combined_mask


def process_and_save_label_mask(npz_data, original_image, output_dir, selected_label):
    """Process and save the mask for the selected label."""
    combined_mask = find_label_masks(npz_data, selected_label)

    if not combined_mask.any():
        print(f"No valid mask found for the label '{selected_label}'.")
        return

    overlay_image = apply_mask(original_image, combined_mask)
    save_masked_image(overlay_image, selected_label, output_dir)

    print(f"Masked image for label '{selected_label}' saved to '{output_dir}'")


def main():
    sam_path = "/home/zeta/Workbenches/ActionGenome_ScenarioCLIP/ActionGenome_DataSet/src/results/sam_results/000000060623_grounding_sam.npz"
    img_path = "/mnt/MIG_store/Datasets/coco/train2017/000000060623.jpg"
    output_dir = "masked_outputs"

    selected_label = "bowl"

    npz_data, original_image = load_data(sam_path, img_path)
    create_output_dir(output_dir)
    process_and_save_label_mask(npz_data, original_image, output_dir, selected_label)


if __name__ == "__main__":
    main()
