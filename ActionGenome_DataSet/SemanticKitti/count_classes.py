import json

with open('classes.json', 'r') as f:
    data = json.load(f)
    
    
actions = len(data['actions'])
relations = len(data['relations'])
objects = len(data['objects'])

print(f"Actions: {actions}")
print(f"Relations: {relations}")
print(f"Objects: {objects}")