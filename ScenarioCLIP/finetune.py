import warnings
warnings.filterwarnings('ignore')

import lightning as L
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from transformers import CLIPTokenizer
from models.pyramid_clip import PyramidCLIP, PyramidCLIP
from models.scenario_clip_1 import ScenarioCLIP1, ScenarioCLIP
from data.datamodule import ActionGenomeDataModule
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

parser = ArgumentParser()

parser.add_argument('--architecture', type=str, default='scenario_clip', help='Architecture of the model to be trained', choices=['pyramid_clip', 'scenario_clip', 'scenario_clip_kd'])
parser.add_argument('--img_dir', type=str, required=True, help='Path to the directory containing the images')
parser.add_argument('--data_dir', type=str, required=True, help='Path to the directory containing the data')
parser.add_argument('--exps_dir', type=str, required=True, help='Path to the directory to store checkpoints, logs and any other files')
parser.add_argument('--metadata_json', type=str, default='./combined.json', help='Path to the metadata JSON file')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloader')
parser.add_argument('--max_epochs', type=int, default=10, help='Number of epochs to train the model')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for training')
parser.add_argument('--vision_model_name', type=str, default='openai/clip-vit-base-patch32', help='Pretrained model for vision')
parser.add_argument('--text_model_name', type=str, default='openai/clip-vit-base-patch32', help='Pretrained model for text')
parser.add_argument('--tokenizer_name', type=str, default='openai/clip-vit-base-patch32', help='Pretrained tokenizer for CLIP')
parser.add_argument('--monitor_device_usage_stats', action='store_true', help='Log device usage statistics at each step and epoch')

args = parser.parse_args()

if __name__ == "__main__":
    time_now = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    data_module = ActionGenomeDataModule(args.img_dir, args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, metadata_json=args.metadata_json)

    if args.architecture == 'pyramid_clip':
        pyramid_clip_model = PyramidCLIP(vision_model_name=args.vision_model_name, text_model_name=args.text_model_name)
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
        model = PyramidCLIP(pyramid_clip_model, tokenizer, lr=args.lr)

    elif args.architecture == 'scenario_clip':
        scenario_clip_model = ScenarioCLIP(vision_model_name=args.vision_model_name, text_model_name=args.text_model_name)
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
        model = ScenarioCLIP1(scenario_clip_model, tokenizer, lr=args.lr, contrastive_only=True)

    elif args.architecture == 'scenario_clip_kd':
        scenario_clip_model = ScenarioCLIP(vision_model_name=args.vision_model_name, text_model_name=args.text_model_name)
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
        model = ScenarioCLIP1(scenario_clip_model, tokenizer, lr=args.lr)

    else:
        raise NotImplementedError(f"The architecture ({args.architecture}) has not been implemented.")

    logger = TensorBoardLogger(save_dir=f"{args.exps_dir}/train", name=f"{args.architecture}/{Path(args.metadata_json).stem}_bs_{args.batch_size}_lr_{args.lr}", version=time_now, sub_dir="tensorboard")

    callbacks_list = []

    if args.monitor_device_usage_stats:
        gpu_stats = DeviceStatsMonitor()
        callbacks_list.append(gpu_stats)

    trainer = L.Trainer(max_epochs=args.max_epochs, devices=1, accelerator="gpu", callbacks=callbacks_list, logger=logger, log_every_n_steps=10)
    trainer.fit(model, datamodule=data_module)
