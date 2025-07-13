storage_path = "/mnt/MIG_archive/Datasets/kinetics-results"
root_path = "/mnt/MIG_store/Datasets/kinetics-400/compress/train_256"

import os
import json
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

chunks = os.listdir(os.path.join(storage_path, "gemma_jsons"))

total_actions = chunks

classes_path = os.path.join(storage_path, "classes.json")
metadata_path = "./metadata.json"

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

        if not os.path.isdir(os.path.join(sub_chunk_path, "frame_0001")):
            continue

        sample = {}

        basename = f"{chunk}/{sub_chunk}"
        gemma_path = os.path.join(
            storage_path, "gemma_jsons", chunk, sub_chunk, "frame_0001_gemma.json"
        )
        dino_path = os.path.join(
            storage_path, "dino_results", chunk, sub_chunk, "frame_0001_grounding_dino.json"
        )

        npz_path = os.path.join(
            storage_path, "sam_results", chunk, sub_chunk, "frame_0001_grounding_sam.npz"
        )

        img_path = os.path.join(root_path, chunk, sub_chunk, "frame_0001.jpg")

        sample['id'] = basename
        sample['image_path'] = img_path

        with open(gemma_path, 'r') as f:
            gemma_data = json.load(f)

        sample['action'] = chunk
        sample["dense_caption"] = gemma_data["response"]["dense caption"]
        sample['relations'] = []
        
        old_og_foc = gemma_data['focused_regions']
        og_foc = {}
        
        for key in old_og_foc.keys():
            if key.endswith("jpg"):
                continue
            og_foc[key] = old_og_foc[key]
            
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
        
        with open(dino_path, 'r') as f:
            dino_data = json.load(f)
            
        for obj in list(sample_objects):
            for idx, lbl in enumerate(dino_data["labels"]):
                if lbl == obj:
                    objects.append([obj, dino_data["boxes"][idx]])
                    break
                
        sample['objects'] = objects
        metadata['data'].append(sample)
    
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=4)
    
with open(classes_path, 'w') as f:
    json.dump(
        {
            'objects': list(total_objects),
            'actions': list(total_actions),
            'relations': list(total_relations)
        }, f, indent=4
    )