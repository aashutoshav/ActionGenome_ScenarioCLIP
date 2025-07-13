storage_path = "./"
root_path = "/scratch/saali/Datasets/Multi_Moments_in_Time/videos"

import os
import json
from tqdm import tqdm

chunks = os.listdir(root_path)

total_actions = chunks

classes_path = os.path.join(storage_path, "classes.json")
metadata_path = os.path.join(storage_path, "metadata.json")

metadata = {
    'data': []
}

total_objects = set()
total_relations = set()

for chunk in tqdm(chunks):
    chunk_path = os.path.join(storage_path, "sam_results", chunk)
    if not os.path.isdir(chunk_path):
        continue

    for sub_chunk in os.listdir(chunk_path):
        sub_chunk_path = os.path.join(chunk_path, sub_chunk)

        if not os.path.isdir(os.path.join(sub_chunk_path)):
            continue

        for f in os.listdir(sub_chunk_path):
            if not os.path.isdir(os.path.join(sub_chunk_path, f)):
                continue

            sample = {}

            basename = f"{chunk}/{sub_chunk}/{f}"
            gemma_path = os.path.join(
                storage_path, "gemma_jsons", chunk, sub_chunk, f + "_gemma.json"
            )
            dino_path = os.path.join(
                storage_path, "dino_results", chunk, sub_chunk, f + "_grounding_dino.json"
            )

            with open(gemma_path, 'r') as f:
                gemma_data = json.load(f)

            with open(dino_path, 'r') as f:
                dino_data = json.load(f)

            sample['id'] = basename
            sample["image_path"] = gemma_data["image"]
            sample['action'] = chunk
            sample['relations'] = []
            sample['dense_caption'] = gemma_data['response']['dense caption']

            og_foc = gemma_data['focused_regions']
            sample_objects = set()

            for x in og_foc.keys():
                pth = x
                rel = og_foc[x]['relation']
                sample['relations'].append([pth, rel])
                total_relations.add(' '.join(rel))

                labels = og_foc[x]['labels']
                total_objects.update(labels)
                sample_objects.update(labels)

            objects = []

            for obj in list(sample_objects):
                for idx, lbl in enumerate(dino_data["labels"]):
                    if lbl == obj:
                        objects.append([obj, dino_data['boxes'][idx]])
                        break

            sample['objects'] = objects
            metadata['data'].append(sample)

with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=4)

with open(classes_path, 'w') as f:
    json.dump({
        'objects': list(total_objects),
        'relations': list(total_relations),
        'actions': total_actions
    }, f, indent=4)
