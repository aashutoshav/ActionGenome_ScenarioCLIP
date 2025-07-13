import json
import warnings

warnings.filterwarnings("ignore")

data = json.load(open("file.json"))

objects = data["objects"]

mapping = {}

import spacy

nlp = spacy.load("en_core_web_md")

object_clusters = {}

similarity_threshold = 0.8

for object in objects:
    object_doc = nlp(object)
    matched = False
    for key in object_clusters:
        key_doc = nlp(key)
        if object_doc.similarity(key_doc) > similarity_threshold:
            object_clusters[key].append(object)
            matched = True
            break
    if not matched:
        object_clusters[object] = [object]

super_class_mapping = {}

for key, group in object_clusters.items():
    for g in group:
        super_class_mapping[g] = key

new_classes = list(set(object_clusters.keys()))

new_mappings = {}

for key, value in super_class_mapping.items():
    for k in mapping.keys():
        if mapping[k] == key:
            new_mappings[k] = value

super_class_mapping.update(new_mappings)

final_dict = {
    "object_classes": new_classes,
    "original_to_new_clusters_mapping": super_class_mapping,
}

with open("object_classes.json", "w") as f:
    json.dump(final_dict, f, indent=4)
