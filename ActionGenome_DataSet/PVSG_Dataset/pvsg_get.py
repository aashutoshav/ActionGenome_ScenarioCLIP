import json
from tqdm import tqdm

with open("JSONS/pvsg_metadata.json", "r") as read_file:
    data = json.load(read_file)

new_output = {
    "data": []
}

metadata = data["data"]

total_actions = set()
total_objects = set()
total_relations = set()

for sample in tqdm(metadata):
    new_sample = {}

    new_sample['id'] = sample['id']
    new_sample["image_path"] = sample["image_path"]
    new_sample['action'] = sample['action']
    total_actions.add(sample['action'])
    new_sample['dense_caption'] = sample['dense_caption']

    new_sample['relations'] = sample['relations']
    new_sample['objects'] = []

    sample_objects = set()

    for rel in sample['relations']:
        reln = rel[1]
        total_relations.add(' '.join(reln))
        if len(reln) > 0:
            sample_objects.add(reln[0])
        if len(reln) > 2:
            sample_objects.add(reln[2])

    og_objects = sample['objects']
    for obj in og_objects:
        obj_name = obj[0]
        obj_box = obj[1]
        
        if obj_name in sample_objects:
            new_sample['objects'].append([obj_name, obj_box]) 
            
        
    total_objects.update(sample_objects)
        
    new_output['data'].append(new_sample)
    
with open("pvsg_metadata.json", 'w') as f:
    json.dump(
        new_output,
        f,
        indent=4,
    )
    
with open("pvsg_classes.json", 'w') as f:
    json.dump(
        {
            "actions": list(total_actions),
            "objects": list(total_objects),
            "relations": list(total_relations)
        }, f, indent=4
    )