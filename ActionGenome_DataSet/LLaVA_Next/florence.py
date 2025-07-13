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
from PIL import Image
from torch.utils.data import DataLoader
from LLaVA_Next.promptTemplate import TEMPLATE_ACTION_OBJECTS_RELATIONS_LIST_FLORENCE
from transformers.image_utils import infer_channel_dimension_format
from fuzzywuzzy import fuzz
from tqdm import tqdm
from dataloaders.coco_dataset import COCO_Dataset, custom_collate_coco
from dataloaders.glamm_dataset import GLAMM_Dataset, custom_collate_glamm
from dataloaders.MMT_dataset import MMT_Dataset, custom_collate_mmt
from transformers import AutoModelForCausalLM, AutoProcessor, LlavaNextProcessor, LlavaNextForConditionalGeneration, AutoTokenizer

# ---------------------------------------------------------------------------- #
#                            LLAVA-NEXT + MISTRAL-7b                           #
# ---------------------------------------------------------------------------- #


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

    def format_labels(self, labels):
        label_count = {}
        formatted_labels = []

        for label in labels:
            if label in label_count:
                label_count[label] += 1
                formatted_labels.append(f"{label}{label_count[label]}")
            else:
                label_count[label] = 1
                formatted_labels.append(label)

        return formatted_labels

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
        device_florence = torch.device("cuda:1")
        device_llava = torch.device("cuda:0")
        device_phi3 = torch.device("cuda:2")

        # ------------------------------ Local Variable ----------------------------- #

        if image_loader is None:
            image_loader = self.create_dataset(args)

        # -------------------------------- Load LLAVA Next Model -------------------------------- #

        florence_model_id = "microsoft/Florence-2-large"
        llava_model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
        phi3_model_id = "microsoft/Phi-3-mini-4k-instruct"

        florence_model = (
            AutoModelForCausalLM.from_pretrained(
                florence_model_id, trust_remote_code=True, torch_dtype="auto"
            )
            .eval()
            .to(device_florence)
        )

        phi3_model = AutoModelForCausalLM.from_pretrained(
            phi3_model_id, trust_remote_code=True, torch_dtype="auto"
        ).eval().to(device_phi3)

        florence_processor = AutoProcessor.from_pretrained(
            florence_model_id, trust_remote_code=True
        )

        phi3_tokenizer = AutoTokenizer.from_pretrained(phi3_model_id)

        PROMPT_TEMPLATE = ["RETURN TRUE IF THE IMAGE HAS AN ACTION IN IT OTHERWISE FALSE"]
        PROMPT = PROMPT_TEMPLATE * args.batch_size

        llava_model = (
            LlavaNextForConditionalGeneration.from_pretrained(
                llava_model_id,
                cache_dir=args.cache_dir,
            )
            .to(device_llava)
            .eval()
        )

        llava_processor = LlavaNextProcessor.from_pretrained(
            llava_model_id, cache_dir=args.cache_dir, pad_token="<pad>", do_rescale=False
        )

        # ------------------------ Generate Responses for Images in batch ------------------------ #
        response_dict = {}
        successCount = 0
        Count = 0
        total_files_skipped = 0

        florence_prompt = "<CAPTION>"

        with tqdm(image_loader) as pbar:
            for img_file, batch_images, hint in pbar:
                batch = len(batch_images)

                for i in range(batch):
                    inputs = florence_processor(
                        text=florence_prompt,
                        images=Image.open(img_file[i]),
                        return_tensors="pt",
                    ).to(device_florence, torch.float16)

                    generated_ids = florence_model.generate(
                        input_ids=inputs["input_ids"].to(device_florence),
                        pixel_values=inputs["pixel_values"].to(device_florence),
                        max_new_tokens=1024,
                        early_stopping=False,
                        do_sample=False,
                        num_beams=3,
                    )

                    generated_text = florence_processor.batch_decode(
                        generated_ids, skip_special_tokens=False
                    )[0]
                    parsed_answer = florence_processor.post_process_generation(
                        generated_text,
                        task=florence_prompt,
                        image_size=(batch_images[i].shape[1], batch_images[i].shape[2]),
                    )

                    generated_florence_caption = parsed_answer[
                        "<CAPTION>"
                    ]

                    phi3_prompt = """
                    <|system|>You are a helpful assistant.<|end|>
                    <|user|>Given the following description, please provide the objects and relations between the objects in the following dictionary format.
                    {
                        'objects': [object1, object2, ..., 'objectn'],
                    }
                    {mdcplaceholder}
                    <|end|>
                    <|assistant|>
                    """

                    phi3_prompt = phi3_prompt.replace("{mdcplaceholder}", generated_florence_caption)

                    phi3_inputs = phi3_tokenizer(
                        [phi3_prompt], return_tensors='pt', padding=True
                    ).to(device_phi3)
                    
                    generated_phi3_ids = phi3_model.generate(
                        **phi3_inputs, max_new_tokens=100, do_sample=True
                    )
                    
                    phi3_response = phi3_tokenizer.batch_decode(
                        generated_phi3_ids, skip_special_tokens=True
                    )[0]

                    response_str = phi3_response.replace("'", '"')
                    phi3_objects = re.search(
                        r'"objects": \[(.*?)\]', response_str, re.DOTALL
                    )

                    filler = f"{hint[i]} + {generated_florence_caption}"
                    PROMPT[i] = PROMPT[i].replace("{hintPlaceholder}", filler)
                    PROMPT[i] = PROMPT[i].replace("{objectsPlaceholder}", f'{phi3_objects}')

                llava_inputs = llava_processor(
                    text=PROMPT[:batch],
                    images=batch_images,
                    return_tensors="pt",
                    padding=True,
                ).to(device_llava)

                with torch.inference_mode():
                    generate_ids = llava_model.generate(**llava_inputs, max_new_tokens=200)

                llava_responses = llava_processor.batch_decode(
                    generate_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                for i in range(batch):

                    Count += 1
                    json_path = img_file[i].replace(
                        ".jpg", "_florence.json"
                    )
                    response = llava_responses[i].split("INST]")[-1].strip()
                    print(f"Response for {img_file[i]} : {response}")

                    try:
                        response = json.loads(
                            response.replace("```json", "").replace("```", "")
                        )
                        response_dict = {
                            "image": img_file[i],
                            "response": response,
                        }
                        with open(json_path, "w") as f:
                            json.dump(response_dict, f)
                            successCount += 1

                    except Exception as e:
                        print(f"Skipping File : {img_file[i]} | Error : {e}")
                        total_files_skipped += 1

        print(f"Successfully Read files {successCount}/{Count}")

        return response_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        # default="/home/zeta/Workbenches/ActionGenome_ScenarioCLIP/ActionGenome_DataSet/assets/GLAMM_dataset", # for GLAMM
        # default="/home/zeta/Workbenches/ActionGenome_ScenarioCLIP/ActionGenome_DataSet/assets/MMT_dataset",  # for MNT
        # default="/home/zeta/Workbenches/ActionGenome_ScenarioCLIP/ActionGenome_DataSet/assets/coco/subtrain2017",  # for coco
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

    vqa_llava = VQA_LLaVA("VQA_LLaVA")
    vqa_llava.generate_response(args)


if __name__ == "__main__":
    main()
