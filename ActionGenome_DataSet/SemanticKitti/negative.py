import json
from tqdm import tqdm
import os

storage_dir = "/scratch/abhijitdas/ACTION_GENOME_RESULTS/KITTI_RESULTS/"

with open("metadata.json", "r") as f:
    metadata = json.load(f)

md = metadata['data']

new_data = []

for sample in tqdm(md):
    impath = sample['image_path']
    chunk = impath.split('/')[-2]
    id = sample['id']

    with open(os.path.join(storage_dir, "gemma_jsons", chunk, id + "_gemma.json"), 'r') as f:
        gemma_data = json.load(f)

    sample['image_path'] = impath + ".png"

    og_objects = gemma_data['response']['objects']
    og_relations = gemma_data["response"]["relations"]
    
    relations = sample['relations']
    
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
        
        updated_relations.append(
            [foc[0], foc[1], neg_rels[:3]]
        )
        
    sample['relations'] = updated_relations
    
    new_data.append(sample)
    
with open("metadata_w_negative.json", "w") as f:
    json.dump({
        "data": new_data
    }, f, indent=4)