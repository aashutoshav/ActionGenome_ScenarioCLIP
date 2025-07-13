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
from dataloaders.cc3m_dataset import CC3M_Dataset, custom_collate_CC3M

torch.cuda.empty_cache()


def create_dataset(args):
    image_size = (
        448,
        448,
    )

    if "cc3m" in args.root_dir.lower():
        image_dataset = CC3M_Dataset(
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
            collate_fn=custom_collate_CC3M,
        )
    else:
        raise NotImplementedError
        
    return image_loader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/media/bitsmig/Expansion/Archive/CC3M/cc3m/cc3m",
    )
    parser.add_argument("--storage_dir", type=str, default="/data/cc3m")
    parser.add_argument("--start_chunk", type=int, default=0)
    parser.add_argument("--end_chunk", type=int, default=331)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--gpu", type=str, default="3")
    args = parser.parse_args()

    d = create_dataset(args)
    
    for _, _ in enumerate(d):
        pass

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if __name__ == "__main__":
    main()