import numpy as np
from tqdm import tqdm
import json
import os

path = "/scratch/saali/Datasets/action-genome/results_9_10/metadata_63k.json"

total_objects = set()
total_actions = set()
total_relations = set()

with open(path, 'r') as f:
    data = json.load(f)

samples = data['data']

new_metadata_json = {
    "data": []
}

for sample in tqdm(samples):
    id = sample['id']
    image_path = sample['image_path']
    action = sample['action']
    dense_caption = sample["dense_caption"]
    relations = sample["relations"]
    bboxes = sample["objects"]

    sample_objects = set()

    pres_dict = {
        "id": id,
        "image_path": image_path,
        "action": action,
        "dense_caption": dense_caption,
        "relations": relations,
        "objects": []
    }

    for foc_reg in relations:
        rel = foc_reg[1]
        if len(rel) > 0:
            sample_objects.add(rel[0])
        if len(rel) > 2:
            sample_objects.add(rel[2])

        total_relations.add(" ".join(rel))

    npz_file = os.path.join("sam_results", id + "_grounding_sam.npz")

    total_objects.update(sample_objects)
    total_actions.add(action)

    npz_data = np.load(npz_file)
    labels = npz_data["labels"]
    
    for x in list(sample_objects):
        for bbox in bboxes:
            if bbox[0] == x:
                pres_dict["objects"].append([
                    x, bbox[1]
                ])
                break

    new_metadata_json['data'].append(pres_dict)

with open("new_metadata_63k.json", 'w') as f:    
    json.dump(new_metadata_json, f, indent=4)

with open('classes.json', 'w') as f:
    json.dump({
        "objects": list(total_objects),
        "actions": list(total_actions),
        "relations": list(total_relations)
    }, f, indent=4)