import json
import warnings
warnings.filterwarnings("ignore")

data = json.load(open('file.json'))

actions = data['actions']

new_actions = set()

mapping = {}

for action in actions:
    if action.endswith('ed'):
        new_actions.add(action[::-1].replace('de', 'gni', 1)[::-1])
        mapping[action] = action[::-1].replace("de", "gni", 1)[::-1]
    else:
        new_actions.add(action)

actions = list(new_actions)

import spacy

nlp = spacy.load("en_core_web_md")

action_clusters = {}

similarity_threshold = 0.8

for action in actions:
    action_doc = nlp(action)
    matched = False
    for key in action_clusters:
        key_doc = nlp(key)
        if action_doc.similarity(key_doc) > similarity_threshold:
            action_clusters[key].append(action)
            matched = True
            break
    if not matched:
        action_clusters[action] = [action]

super_class_mapping = {}

for key, group in action_clusters.items():
    for g in group:
        super_class_mapping[g] = key
        
new_classes = list(set(action_clusters.keys()))

new_mappings = {}

for key, value in super_class_mapping.items():
    for k in mapping.keys():
        if mapping[k] == key:
            new_mappings[k] = value

super_class_mapping.update(new_mappings)

final_dict = {
    "action_classes": new_classes,
    "original_to_new_clusters_mapping": super_class_mapping
}

with open("action_classes.json", "w") as f:
    json.dump(final_dict, f, indent=4)