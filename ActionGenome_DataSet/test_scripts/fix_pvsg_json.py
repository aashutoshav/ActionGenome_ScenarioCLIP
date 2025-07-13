import json
from tqdm import tqdm

path = "metadata_w_neg.json"

new_data = {
    'data': []
}

with open(path, 'r') as f:
    data = json.load(f)
    
for sample in tqdm(data):
    new_sample = {}
    
    new_sample['id'] = sample['id']
    new_sample['image_path'] = '/'.join(x for x in sample['image_path'])
    new_sample['action'] = sample['action']
    new_sample["dense_caption"] = sample["dense caption"]
    new_sample['objects'] = sample['objects']
    
    old_rels = sample['relations']
    
    new_rels = []
    
    for key in old_rels.keys():
        foc_path = key
        rel = old_rels[key]['relation']
        neg_samples = [x for x in old_rels[key]['neg_samples']]
        
        new_rels.append(
            [foc_path, rel, neg_samples]
        )
        
    new_sample['relations'] = new_rels
    
    new_data['data'].append(new_sample)
    
with open('metadata_w_neg_fixed.json', 'w') as f:
    json.dump(new_data, f, indent=4)