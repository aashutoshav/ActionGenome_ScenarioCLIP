'''
Making the training and validation chunks for the OpenPSG Dataset
'''

import os
import json
import argparse
import math


def main():
    with open(f"{args.root_folder}/annotations/captions_{args.split}2017.json") as f:
        data = json.load(f)

    results = {}
    entries_in_json = 0
    img_names = []

    for dic in data["images"]:
        filename = dic["file_name"]
        img_id = dic["id"]
        caption = ""
        results[filename] = {"image_id": img_id, "caption": caption}
        entries_in_json += 1
        img_names.append(filename)

    for dic in data["annotations"]:
        filename = "000000" + str(dic["image_id"]) + ".jpg"
        if filename not in results:
            continue
        caption = dic["caption"]
        results[filename]["caption"] += caption

    with open(os.path.join(args.storage_dir, args.out_details_json_path), "w") as f:
        json.dump(results, f, indent=4)

    total_images = entries_in_json
    chunk_size = math.ceil(total_images / 30)
    chunks = {}

    for i in range(30):
        start_index = i * chunk_size
        end_index = min((i + 1) * chunk_size, total_images)
        chunks[f"chunk_{i}"] = img_names[start_index:end_index]

    train_chunks = {str(args.split): chunks}
    with open(os.path.join(args.storage_dir, args.out_chunks_json_path), "w") as f:
        json.dump(train_chunks, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_folder",
        type=str,
        # default="/mnt/MIG_store/Datasets/coco",
        default="/mnt/MIG_store/Datasets/coco",
    )
    parser.add_argument(
        "--storage_dir",
        type=str,
        default="./",
    )
    parser.add_argument(
        "--out_details_json_path",
        type=str,
        default="./val_details.json",
    )
    parser.add_argument(
        "--out_chunks_json_path",
        type=str,
        default="./val_chunks.json",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="val",
    )

    args = parser.parse_args()
    main()
