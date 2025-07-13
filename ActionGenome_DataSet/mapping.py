import json
import os
import random
import re
import argparse

def generate_random_color():
    return [random.randint(0, 255) for _ in range(3)]


def normalize_name(name):
    name = re.sub(r"\s+\d+$", "", name.strip().lower())
    return " ".join(name.split())


def main():
    objects_list = []
    actions_list = []
    relations_list = []
    object_colors = {}

    for filename in os.listdir(args.path_to_jsons):
        if "llama" in filename:
            filepath = os.path.join(args.path_to_jsons, filename)

            with open(filepath, "r") as f:
                data = json.load(f)

                objects = data.get("response", {}).get("objects", [])
                actions = data.get("response", {}).get("action", None)
                relations = data.get("response", {}).get("relations", [])

                normalized_objects = [normalize_name(obj) for obj in objects]
                objects_list.extend(normalized_objects)

                if actions and actions != "None":
                    actions_list.append(actions)

                relations_list.extend(relations)

    unique_objects = {normalize_name(obj) for obj in objects_list}
    for obj in unique_objects:
        object_colors[obj] = generate_random_color()

    with open(args.colorMapping_path, "w") as f:
        json.dump(object_colors, f, indent=4)

    output = {
        "activities": actions_list,
        "objects": objects_list,
        "relations": relations_list,
    }

    with open(args.labels_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Labels and color mapping saved to {args.labels_path} and {args.colorMapping_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_llama_jsons", type=str, default="/home/zeta/Workbenches/ActionGenome_ScenarioCLIP/ActionGenome_DataSet/assets/coco/results")
    parser.add_argument(
        "--colorMapping_path",
        type=str,
        default="/home/zeta/Workbenches/ActionGenome_ScenarioCLIP/ActionGenome_DataSet/dataset/color_mapping.json",
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        default="/home/zeta/Workbenches/ActionGenome_ScenarioCLIP/ActionGenome_DataSet/dataset/labels.json",
    )
    args = parser.parse_args()
    main()