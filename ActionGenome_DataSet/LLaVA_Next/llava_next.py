import re
import os
import sys

sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-1]))

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import re
import os
import json
import torch
import argparse
import torch.nn as nn
import requests
from torch.utils.data import DataLoader
from LLaVA_Next.promptTemplate import TEMPLATE_1, TEMPLATE_ACTION_OBJECTS_RELATIONS_LIST_LLAVA
from transformers.image_utils import infer_channel_dimension_format
from fuzzywuzzy import fuzz
from tqdm import tqdm
from dataloaders.coco_dataset import COCO_Dataset, custom_collate_coco
from dataloaders.glamm_dataset import GLAMM_Dataset, custom_collate_glamm
from dataloaders.MMT_dataset import MMT_Dataset, custom_collate_mmt
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

import torch
torch.cuda.empty_cache

# ---------------------------------------------------------------------------- #
#                            LLAVA-NEXT + MISTRAL-7b                           #
# ---------------------------------------------------------------------------- #


MODEL_NAME_DICT = {
    "llava_mistral": "llava-hf/llava-v1.6-mistral-7b-hf",
    "llava_34b": "llava-hf/llava-v1.6-34b-hf",
}


class VQA_LLaVA:
    def __init__(self, name):
        self.className = str(name)
        return

    def get_unique_names(self, input_string, similarity_threshold=99):
        # Split the input string into items separated by commas
        items = input_string.split(", ")

        # Initialize an empty list to store unique names
        unique_names = []

        # Regular expression pattern to match and remove the leading numbers and punctuation
        pattern = r"^\d+\.\s*(.*)"

        # Iterate over each item and extract the names
        for item in items:
            match = re.match(pattern, item)
            if match:
                name = match.group(1).strip()
                # Check if the name is similar to any existing unique names
                similar = False
                for unique_name in unique_names:
                    similarity = fuzz.token_sort_ratio(name, unique_name)
                    if similarity >= similarity_threshold:
                        similar = True
                        break
                # If not similar, add to unique_names
                if not similar:
                    unique_names.append(name)

        return unique_names

    def create_dataset(self, args):
        # --------------------- Load Image using Dataloader -------------------- #
        image_size = (
            336,
            336,
        )  # Every image is downscaled/upscaled to this dimension for batch processing

        if "glamm" in args.root_dir.lower():
            image_dataset = GLAMM_Dataset(
                root_dir=args.root_dir,
                start_chunk=args.start_chunk,
                end_chunk=args.end_chunk,
                image_shape=image_size,
            )

            image_loader = DataLoader(
            image_dataset,
            batch_size=args.batch_size,
            num_workers=args.max_workers,
            shuffle=False,
            collate_fn=custom_collate_glamm,
            )
        elif "mmt" in args.root_dir.lower():
            image_dataset = MMT_Dataset(
                root_dir=args.root_dir,
                start_chunk=args.start_chunk,
                end_chunk=args.end_chunk,
                image_shape=image_size,
            )

            image_loader = DataLoader(
            image_dataset,
            batch_size=args.batch_size,
            num_workers=args.max_workers,
            shuffle=False,
            collate_fn=custom_collate_mmt,
        )
        elif "coco" in args.root_dir.lower():
            image_dataset = COCO_Dataset(
                root_dir=args.root_dir,
                start_chunk=args.start_chunk,
                end_chunk=args.end_chunk,
                image_shape=image_size,
            )

            image_loader = DataLoader(
            image_dataset,
            batch_size=args.batch_size,
            num_workers=args.max_workers,
            shuffle=False,
            collate_fn=custom_collate_coco,
        )
        else:
            raise NotImplementedError

        return image_loader

    def generate_response(self, args, image_loader=None):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # ------------------------------ Local Variable ----------------------------- #

        MODEL_NAME = MODEL_NAME_DICT[f"llava_mistral"]
        PROMPT_TEMPLATE = [TEMPLATE_ACTION_OBJECTS_RELATIONS_LIST_LLAVA]

        # add folder name to the prompt for mmt

        if image_loader is None:
            image_loader = self.create_dataset(args)

        PROMPT = PROMPT_TEMPLATE * args.batch_size

        # -------------------------------- Load LLAVA Next Model -------------------------------- #
        processor = LlavaNextProcessor.from_pretrained(
            MODEL_NAME, cache_dir=args.cache_dir, pad_token="<pad>", do_rescale=False
        )

        model = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            cache_dir=args.cache_dir,
        )
        model.to(device)
        
        model.eval()

        # ------------------------ Generate Responses for Images in batch ------------------------ #
        response_dict = {}
        successCount = 0
        Count = 0
        total_files_skipped = 0

        summary_json_path = os.path.join(args.root_dir, "summary_annotations.json")
        summary_data = []

        with tqdm(image_loader) as pbar:
            for img_file, batch_images, hint in pbar:
                batch = len(batch_images)

                for i in range(batch):
                    filler = f"You are given the following hint. You can use this hint to estimate the value of the relation key of the json you will generate, but you can't copy it exactly: {hint[i]}"
                    filler = filler + f"DO NOT USE COLLECTIVE NOUNS, use [person 1, person 2] INSTEAD OF [some people]"
                    PROMPT[i] = PROMPT[i].replace("{hintPlaceholder}", filler)

                inputs = processor(
                    text=PROMPT[:batch],
                    images=batch_images,
                    return_tensors="pt",
                    padding=True,
                ).to(device)

                with torch.inference_mode():
                    generate_ids = model.generate(**inputs, max_new_tokens=200)

                responses = processor.batch_decode(
                    generate_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                # responses = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat", force_download=True).batch_decode(
                #     model.config, generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                # )

                for i in range(batch):
                    Count += 1
                    json_path = img_file[i].replace(".jpg", "_llava_next.json") # for GLAMM and MMT
                    # json_path = img_file[i].replace(".png", "_llava_next.json") # for coco
                    response = responses[i].split("INST]")[-1].strip()
                    # print(f"Response : {response}")

                    try:
                        response = json.loads(response.replace("```json", "").replace("```",""))
                        response_dict = {
                            "image": img_file[i],
                            "response": response,
                        }
                        with open(json_path, "w") as f:
                            json.dump(response_dict, f)
                            successCount += 1

                        relations = response_dict.get("response", {}).get("relations", [])

                        summary_entry = {
                            "image_id": img_file[i],
                            'ground-truth-caption': hint[i],
                            'generated_relations': relations 
                        }
                        summary_data.append(summary_entry)

                    except Exception as e:
                        print(f"Skipping File : {img_file[i]} | Error : {e}")
                        total_files_skipped += 1

        with open(summary_json_path, "w") as f:
            json.dump(summary_data, f)

        print(f"Successfully Read files {successCount}/{Count}")

        return response_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        # default="/home/zeta/Workbenches/ActionGenome_ScenarioCLIP/ActionGenome_DataSet/assets/GLAMM_dataset", # for GLAMM
        # default="/home/zeta/Workbenches/ActionGenome_ScenarioCLIP/ActionGenome_DataSet/assets/MMT_dataset",  # for MNT
        default="/home/zeta/Workbenches/ActionGenome_ScenarioCLIP/ActionGenome_DataSet/assets/coco/checktrain",  # for coco
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--cache_dir",
        default="/home/zeta/Workbenches/ActionGenome_ScenarioCLIP/ActionGenome_DataSet/assets/.cache",
    )
    parser.add_argument("--start_chunk", type=int, default=-1)
    parser.add_argument("--end_chunk", type=int, default=-1)
    parser.add_argument("--max_workers", type=int, default=8)
    args = parser.parse_args()

    vqa_llava=VQA_LLaVA("VQA_LLaVA")
    vqa_llava.generate_response(args)


if __name__ == "__main__":
    main()
