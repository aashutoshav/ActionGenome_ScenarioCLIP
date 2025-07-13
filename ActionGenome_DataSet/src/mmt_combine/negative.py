import os
import json
from tqdm import tqdm

storage_path = "./MMT_results"
metadata_path = "./metadata.json"

with open(metadata_path, 'r') as f:
    metadata = json.load(f)

md = metadata['data']
new_data = []

for sample in tqdm(md):
    impath = sample['image_path']
    chunk = sample['action']

    sub_chunk = impath.split('/')[-2]
    basename = impath.split('/')[-1]

    with open(os.path.join(storage_path, "gemma_jsons", chunk, sub_chunk, basename.split('.')[0] + "_gemma.json"), 'r') as f:
        gemma_data = json.load(f)

    og_objects = gemma_data["response"]["objects"]
    og_relations = gemma_data["response"]["relations"]

    relations = sample["relations"]

    updated_relations = []

    for foc in relations:
        foc_img = foc[0]
        rel = foc[1]

        neg_rels = []

        while len(neg_rels) < 3:
            for og_obj in og_objects:
                if [og_obj, rel[1], rel[2]] not in og_relations:
                    neg_rels.append([og_obj, rel[1], rel[2]])
                if [rel[0], rel[1], og_obj] not in og_relations:
                    neg_rels.append([rel[0], rel[1], og_obj])

        updated_relations.append([foc[0], foc[1], neg_rels[:3]])

    sample["relations"] = updated_relations

    new_data.append(sample)

with open("metadata_w_negative.json", "w") as f:
    json.dump({
        "data": new_data
    }, f, indent=4)