path = "/mnt/MIG_archive/Datasets/action-genome-dataset/cc3m/train"
root_dir = "/mnt/MIG_archive/Datasets/cc3m/train/cc3m"

import os
import json
from tqdm import tqdm
import numpy as np

chunks = [str(i) for i in range(332)]
chunks = [chunk.zfill(5) for chunk in chunks]

classes_path = "/home/zeta/Workbenches/clip_jsons/CC3M/classes.json"
metadata_path = "/home/zeta/Workbenches/clip_jsons/CC3M/metadata.json"

metadata = {
    'data': []
}

total_actions = set()
total_objects = set()
total_relations = set()

for chunk in chunks:
    chunk_path = os.path.join(path, "sam_results", chunk)
    if not os.path.exists(chunk_path):
        continue
    if not os.path.isdir(chunk_path):
        continue
    for f in tqdm(os.listdir(chunk_path)):
        if not os.path.isdir(os.path.join(chunk_path, f)):
            continue

        sample = {}

        basename = f
        gemma_path = os.path.join(path, "gemma_jsons", chunk, basename + "_gemma.json")
        dino_path = os.path.join(
            path, "dino_results", chunk, basename + "_grounding_dino.json"
        )
        sam_path = os.path.join(
            path, "sam_results", chunk, basename + "_grounding_sam.npz"
        )

        try:
            with open(gemma_path, "r") as f:
                gemma_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Skipping {gemma_path}: {e}")
            continue

        try:
            with open(dino_path, "r") as f:
                dino_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Skipping {dino_path}: {e}")
            continue

        try:
            npz_data = np.load(sam_path)
            labels = npz_data["labels"]
        except FileNotFoundError as e:
            print(f"Skipping {sam_path}: {e}")
            continue

        img_path = os.path.join(root_dir, chunk, basename+".jpg")
        id = basename
        sample["id"] = id
        sample["image_path"] = img_path
        # with open(gemma_path, "r") as f:
        #     gemma_data = json.load(f)

        sample["action"] = gemma_data["response"]["action"]
        total_actions.add(sample["action"])

        sample["dense_caption"] = gemma_data["response"]["dense caption"]
        sample["relations"] = []
        
        if not "focused_regions" in gemma_data.keys():
            continue

        sample_objects = set()

        for x in gemma_data["focused_regions"].keys():
            pth = x
            rel = gemma_data["focused_regions"][x]["relation"]

            sample["relations"].append([pth, rel])
            total_relations.add(" ".join(rel))

            if len(rel) > 0:
                total_objects.add(rel[0])
                sample_objects.add(rel[0])

            if len(rel) > 2:
                total_objects.add(rel[2])
                sample_objects.add(rel[2])

        objects = []

        # with open(dino_path, "r") as f:
            # dino_data = json.load(f)

        # npz_data = np.load(sam_path)
        labels = npz_data["labels"]

        for obj in list(sample_objects):
            for idx, lbl in enumerate(dino_data["labels"]):
                if lbl == obj:
                    objects.append([obj, dino_data["boxes"][idx]])
                    break

        sample["objects"] = objects

        metadata["data"].append(sample)


with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)

with open("classes.json", "w") as f:
    json.dump(
        {
            "objects": list(total_objects),
            "actions": list(total_actions),
            "relations": list(total_relations),
        },
        f,
        indent=4,
    )
