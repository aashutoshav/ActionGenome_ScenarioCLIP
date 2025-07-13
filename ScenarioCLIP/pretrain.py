import warnings
warnings.filterwarnings('ignore')

import os
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import DeviceStatsMonitor
from models.pyramid_clip import PyramidCLIP
from models.scenario_clip_0 import ScenarioCLIP0
from models.scenario_clip_1 import ScenarioCLIP1
from models.scenario_clip_2 import ScenarioCLIP2
from data.datamodule import ActionGenomeDataModule
from argparse import ArgumentParser
from datetime import datetime

parser = ArgumentParser()

parser.add_argument('--architecture', type=str, default='scenario_clip', help='Architecture of the model to be trained', choices=['pyramid_clip', 'scenario_clip_0', 'scenario_clip_1', 'scenario_clip_2'])
parser.add_argument('--save_dir', type=str, default=f"/scratch/{os.environ.get('USER')}/Exps/scenario-clip", help='Path to the directory to store checkpoints, logs and any other files')
parser.add_argument('--metadata_dir', type=str, default='./metadata', help='Path to the directory containing metadata JSONs')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for dataloader')
parser.add_argument('--max_epochs', type=int, default=10, help='Number of epochs to train the model')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for training')
parser.add_argument('--vision_model_name', type=str, default='openai/clip-vit-base-patch32', help='Pretrained model for vision')
parser.add_argument('--text_model_name', type=str, default='openai/clip-vit-base-patch32', help='Pretrained model for text')
parser.add_argument('--tokenizer_name', type=str, default='openai/clip-vit-base-patch32', help='Pretrained tokenizer for CLIP')
parser.add_argument('--monitor_device_usage_stats', action='store_true', help='Log device usage statistics at each step and epoch')

args = parser.parse_args()

if __name__ == "__main__":
    time_now = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    data_module = ActionGenomeDataModule(metadata_dir=args.metadata_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    if args.architecture == 'pyramid_clip':
        model = PyramidCLIP(args.vision_model_name, args.text_model_name, args.tokenizer_name, lr=args.lr)

    elif args.architecture == 'scenario_clip_0':
        model = ScenarioCLIP0(args.vision_model_name, args.text_model_name, args.tokenizer_name, lr=args.lr)

    elif args.architecture == 'scenario_clip_1':
        model = ScenarioCLIP1(args.vision_model_name, args.text_model_name, args.tokenizer_name, lr=args.lr)

    elif args.architecture == 'scenario_clip_2':
        model = ScenarioCLIP2(args.vision_model_name, args.text_model_name, args.tokenizer_name, lr=args.lr)

    else:
        raise NotImplementedError(f"The architecture ({args.architecture}) has not been implemented.")

    logger = CSVLogger(save_dir=f"{args.save_dir}/pretrain", name=f"{args.architecture}/bs_{args.batch_size}_lr_{args.lr}", version=time_now)

    print(f"Logging experiment to: {args.save_dir}/pretrain/{args.architecture}/bs_{args.batch_size}_lr_{args.lr}/{time_now}")

    callbacks_list = []

    if args.monitor_device_usage_stats:
        gpu_stats = DeviceStatsMonitor()
        callbacks_list.append(gpu_stats)

    trainer = L.Trainer(max_epochs=args.max_epochs, devices=1, accelerator="gpu", callbacks=callbacks_list, logger=logger, log_every_n_steps=10, gradient_clip_algorithm="norm", gradient_clip_val=1.)
    trainer.fit(model, datamodule=data_module)
