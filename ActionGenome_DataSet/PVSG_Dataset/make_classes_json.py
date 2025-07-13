import json

# with open("classes.json", 'r') as f:
#     data = json.load(f)
    
# with open("metadata.json", 'r') as f:
#     metadata = json.load(f)
    
# relations = {
#     "relations": []
# }

# for sample in metadata:
#     for foc_img in sample['relations']:
#         rel = sample['relations'][foc_img]['relation']
        
#         relations["relations"].append(' '.join(rel))
        
# relations["relations"] = list(set(relations["relations"]))

# with open("tmp.json", 'w') as f:
#     json.dump({
#         "actions": data["actions"],
#         "objects": data["objects"],
#         "relations": relations["relations"]
#     }, f, indent=4)

with open("classes.json", 'r') as f:
    data = json.load(f)
    
new_objects = []

for obj in data['objects']:
    if obj.startswith("#"):
        continue
    new_objects.append(obj)
    
data['objects'] = new_objects
with open("classes.json", 'w') as f:
    json.dump(data, f, indent=4)