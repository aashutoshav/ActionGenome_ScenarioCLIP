import numpy as np
import os
from tqdm import tqdm
import json

metadata_path = "metadata.json"
with open(metadata_path) as f:
    samples = json.load(f)

action_labels = set()
for sample in tqdm(samples, desc="Getting action labels"):
    action_labels.add(sample['action'])

print(f"Total action labels: {len(action_labels)}")

obj_labels = set()
for sample in tqdm(samples, desc="Getting labels"):
    focused_path = list(sample['relations'].keys())[0]
    fopa = '/'.join(focused_path.split("/")[:-1])
    sam_path = fopa + "_grounding_sam.npz"
    if not os.path.exists(sam_path):
        continue
    labels = set(np.load(sam_path)["labels"])

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

print(f"Total object labels: {len(obj_labels)}")

with open('classes.json', 'w') as f:
    json.dump({"actions": list(action_labels), "objects": list(obj_labels)}, f)