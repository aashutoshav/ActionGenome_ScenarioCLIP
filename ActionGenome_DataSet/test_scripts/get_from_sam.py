import numpy as np
import os
from tqdm import tqdm
import json

metadata_path = "metadata_63k.json"
with open(metadata_path) as f:
    metadata = json.load(f)

samples = metadata['data']

obj_labels = set()
for sample in tqdm(samples, desc="Getting labels"):
    id = sample['id']
    sam_path = f"sam_results/{id}_grounding_sam.npz"
    if not os.path.exists(sam_path):
        continue
    labels = set(np.load(sam_path)['labels'])

    label_set = set()

    for label in labels:
        if "another" in label:
            continue
        if any(char.isdigit() for char in label):
            continue
        if len(label.split(" ")) > 1:
            if len(set(label.split(" "))) != len(label.split(" ")):
                continue
        if label.endswith("s") and label[:-1] in obj_labels:
            continue
        if label.endswith("es") and label[:-2] in obj_labels:
            continue
        label_set.add(label)

    obj_labels = obj_labels.union(label_set)

print(f"Total labels: {len(obj_labels)}")

with open("temp_objects.json", 'w') as f:
    json.dump(list(obj_labels), f, indent=4)

with open('temp_objects.json', 'r') as f:
    obj_labels = json.load(f)

path = "classes_63k.json"
with open(path, "r") as f:
    data = json.load(f)

action_classes = data["actions"]
ans_dict = {"actions": action_classes, "objects": []}

obj_labels = [obj for obj in obj_labels if not obj.startswith('#')]

ans_dict["objects"] = obj_labels
with open("new_classes_63k.json", "w") as f:
    json.dump(ans_dict, f, indent=4)
