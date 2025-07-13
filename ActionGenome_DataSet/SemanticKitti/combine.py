path = "/scratch/abhijitdas/ACTION_GENOME_RESULTS/KITTI_RESULTS"
root_dir = "/scratch/abhijitdas/Datasets/KITTI2015/training"

import json
import os
import numpy as np
from tqdm import tqdm

chunks = ['image_2', 'image_3']

classes_path = "/scratch/abhijitdas/ACTION_GENOME_RESULTS/KITTI_RESULTS/classes.json"
metadata_path = "metadata.json"

metadata = {
    'data': []
}

total_objects = set()
total_actions = set()
total_relations = set()

for chunk in chunks:
    chunk_path = os.path.join(path, "sam_results", chunk)
    for f in tqdm(os.listdir(chunk_path)):
        print(f"Processing {f}")
        if not os.path.isdir(os.path.join(chunk_path, f)):
            continue

        sample = {}

        basename = f
        gemma_path = os.path.join(path, "gemma_jsons", chunk, basename+"_gemma.json")
        dino_path = os.path.join(path, "dino_results", chunk, basename+"_grounding_dino.json")
        sam_path = os.path.join(path, "sam_results", chunk, basename+"_grounding_sam.npz")

        img_path = os.path.join(root_dir, chunk, basename)
        
        id = basename
        sample['id'] = id
        sample['image_path'] = img_path
        with open(gemma_path, 'r') as f:
            gemma_data = json.load(f)

        sample['action'] = gemma_data['response']['action']
        total_actions.add(sample['action'])

        sample['dense_caption'] = gemma_data['response']['dense caption']
        sample['relations'] = []

        sample_objects = set()

        for x in gemma_data['focused_regions'].keys():
            pth = x
            rel = gemma_data['focused_regions'][x]['relation']

            sample['relations'].append([pth, rel])
            total_relations.add(' '.join(rel))

            if len(rel) > 0:
                total_objects.add(rel[0])
                sample_objects.add(rel[0])

            if len(rel) > 2:
                total_objects.add(rel[2])
                sample_objects.add(rel[2])

        objects = []
        
        with open(dino_path, 'r') as f:
            dino_data = json.load(f)
            
        npz_data = np.load(sam_path)
        labels = npz_data["labels"]
        
        for obj in list(sample_objects):
            for idx, lbl in enumerate(dino_data["labels"]):
                if lbl == obj:
                    objects.append([obj, dino_data["boxes"][idx]])
                    break
                
        sample['objects'] = objects
        
        metadata['data'].append(sample)
        

with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=4)
    
with open("classes.json", 'w') as f:
    json.dump({
        'objects': list(total_objects),
        'actions': list(total_actions),
        'relations': list(total_relations)
    }, f, indent=4)