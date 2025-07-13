import json
from tqdm import tqdm
import os

datasets = ["vidor", "epic_kitchen"]

final_output = []

for dataset in datasets:
    dataset_path = os.path.join("./", dataset)
    gemma_jsons_path = os.path.join(dataset_path, "gemma_jsons")
    dino_results_path = os.path.join(dataset_path, "dino_results")
    sam_results_path = os.path.join(dataset_path, "sam_results")

    for chunk in tqdm(os.listdir(sam_results_path)):
        chunk_path = os.path.join(sam_results_path, chunk)
        for file in os.listdir(chunk_path):
            file_path = os.path.join(chunk_path, file)
            if not os.path.isdir(file_path):
                continue

            id = os.path.basename(file_path)

            focused_regions = os.listdir(file_path)

            gemma_file = os.path.join(gemma_jsons_path, chunk, id + "_gemma.json")
            dino_file = os.path.join(dino_results_path, chunk, id + "_grounding_dino.json")

            with open(gemma_file, "r") as f:
                gemma_data = json.load(f)

            with open(dino_file, 'r') as f:
                dino_data = json.load(f)

            image_path = gemma_data['image'].split('/')[-4:]

            objects = []

            for i in range(len(dino_data['labels'])):
                objects.append(
                    [
                        dino_data['labels'][i],
                        dino_data['boxes'][i]
                    ]
                )
                
            if "dense caption" in gemma_data["response"]:
                dc = gemma_data["response"]["dense caption"]
            elif "dense_caption" in gemma_data["response"]:
                dc = gemma_data["response"]["dense_caption"]
            else:
                dc = ""

            new_dict = {
                "id": id,
                "image_path": image_path,
                "action": gemma_data["response"]["action"].lower(),
                "dense caption": dc,
                "objects": objects,
                "relations": gemma_data["focused_regions"],
            }

            final_output.append(new_dict)
            
json.dump(final_output, open("metadata.json", "w"), indent=4)