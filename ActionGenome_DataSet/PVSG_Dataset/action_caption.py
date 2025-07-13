import re
import os
import sys
import time

sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-1]))

import json
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloaders.vidor_dataset import custom_collate_vidorForActionCaption, VidOrForActionCaption_Dataset
from dataloaders.epic_kitchen_dataset import EpicForActionCaption_Dataset, custom_collate_epicForActionCaption
from transformers import AutoModelForCausalLM

torch.cuda.empty_cache()


GEMMA_PROMPT = """
    Return only the following information:
    1. Action should be a single word of a broad class, like "playing", "eating", "running", "talking", "driving" etc. Avoid specific actions like "serving", "gulping", "steering" etc.
    4. The description tag should be one or two sentences long, describing the image in general terms.
    Do not provide any extra information or descriptions.
    Give the output in a json format, example:
    {
        "action": "<some action>"
        "dense caption": "The image shows a... <description>"
    }
"""

class VQA_LLaVA:
    def __init__(self, name):
        self.className = str(name)
        return

    def create_dataset(self, args):
        # --------------------- Load Image using Dataloader -------------------- #
        image_size = (
            448,
            448,
        )
        if "vidor" in args.root_dir.lower():
            image_dataset = VidOrForActionCaption_Dataset(
                root_dir=args.root_dir,
                storage_dir=args.storage_dir,
                start_chunk=args.start_chunk,
                end_chunk=args.end_chunk,
                image_shape=image_size,
            )

            image_loader = DataLoader(
                image_dataset,
                batch_size=args.batch_size,
                num_workers=args.max_workers,
                shuffle=False,
                collate_fn=custom_collate_vidorForActionCaption,
            )

        elif 'epic' in args.root_dir.lower():
            image_dataset = EpicForActionCaption_Dataset(
                root_dir=args.root_dir,
                storage_dir=args.storage_dir,
                start_chunk=args.start_chunk,
                end_chunk=args.end_chunk,
                image_shape=image_size,
            )
            
            image_loader = DataLoader(
                image_dataset,
                batch_size=args.batch_size,
                num_workers=args.max_workers,
                shuffle=False,
                collate_fn=custom_collate_epicForActionCaption,
            )

        else:
            raise NotImplementedError

        return image_loader

    def generate_response(self, args, image_loader=None):

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        model = AutoModelForCausalLM.from_pretrained(
            "AIDC-AI/Ovis1.5-Gemma2-9B",
            torch_dtype=torch.bfloat16,
            multimodal_max_length=8192,
            trust_remote_code=True,
        ).cuda()

        text_tokenizer = model.get_text_tokenizer()
        visual_tokenizer = model.get_visual_tokenizer()
        conversation_formatter = model.get_conversation_formatter()

        PROMPT_TEMPLATE = [GEMMA_PROMPT]

        image_loader = self.create_dataset(args)

        PROMPT = PROMPT_TEMPLATE * args.batch_size

        response_dict = {}
        successCount = 0
        Count = 0
        total_files_skipped = 0

        os.makedirs(args.storage_dir, exist_ok=True)
        os.makedirs(os.path.join(args.storage_dir, "gemma_jsons"), exist_ok=True)

        for img_file, batch_images, json_path in tqdm(image_loader):
            batch = len(batch_images)

            for i in range(batch):
                try:
                    query = f"<image>\n{PROMPT[i]}"

                    json_path = json_path[i]

                    with open(json_path, "r") as f:
                        gemma_data = json.load(f)
                        
                    if gemma_data['response']['action'] != '':
                        continue

                    prompt, input_ids = conversation_formatter.format_query(query)
                    input_ids = torch.unsqueeze(input_ids, dim=0).to(
                        device=model.device
                    )
                    attention_mask = torch.ne(
                        input_ids, text_tokenizer.pad_token_id
                    ).to(device=model.device)
                    pixel_values = [
                        visual_tokenizer.preprocess_image(batch_images[i]).to(
                            dtype=visual_tokenizer.dtype, device=visual_tokenizer.device
                        )
                    ]

                    with torch.inference_mode():
                        gen_kwargs = dict(
                            max_new_tokens=1024,
                            do_sample=False,
                            top_p=None,
                            top_k=None,
                            temperature=None,
                            repetition_penalty=None,
                            eos_token_id=model.generation_config.eos_token_id,
                            pad_token_id=text_tokenizer.pad_token_id,
                            use_cache=True,
                        )
                        output_ids = model.generate(
                            input_ids,
                            pixel_values=pixel_values,
                            attention_mask=attention_mask,
                            **gen_kwargs,
                        )[0]
                        output = text_tokenizer.decode(
                            output_ids, skip_special_tokens=True
                        )

                    Count += 1

                    response = json.loads(
                        output.replace("```json", "").replace("```", "")
                    )

                    response_dict = {
                        "image": img_file[i],
                        "response": response,
                    }
                    
                    gemma_data['response']['action'] = response_dict['response']['action']
                    gemma_data['response']['dense caption'] = response_dict['response']['dense caption']

                    with open(json_path, "w") as f:
                        json.dump(gemma_data, f, indent=4)
                        successCount += 1
                        print(f"Saved File : {json_path}", flush=True)

                except Exception as e:
                    print(f"Skipping File : {img_file[i]} | Error : {e}", flush=True)
                    total_files_skipped += 1

        print(f"Successfully Read files {successCount}/{Count}")
        print(f"Total Files Skipped : {total_files_skipped}")

        return response_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/scratch/saali/Datasets/OpenPVSG/data/vidor",
    )
    parser.add_argument(
        "--storage_dir",
        type=str,
        default="/scratch/saali/Datasets/pvsg_results/vidor",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--cache_dir",
        default="/scratch/saali/Datasets/.cache",
    )
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--start_chunk", type=int, default=0)
    parser.add_argument("--end_chunk", type=int, default=400, help="Max is 400 for epic, 288 for vidor")
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()

    os.environ["TRANSFORMERS_CACHE"] = args.cache_dir

    vqa_llava = VQA_LLaVA("VQA_LLaVA")
    vqa_llava.generate_response(args)


if __name__ == "__main__":
    main()
