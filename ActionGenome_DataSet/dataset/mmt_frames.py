import os
from tqdm import tqdm
import subprocess

def get_frames(video_path):
    folder_path = video_path.split(".")[0]
    os.makedirs(folder_path, exist_ok=True)

    fps=1
    
    command = [
        "ffmpeg",
        "-i",
        video_path,
        "-vf",
        f"fps={fps}",
        f"{folder_path}/%04d.png",
    ]
    
    try:
        subprocess.run(command, check=True)
        
    except Exception as e:
        pass

def process_mmt(root_dir):
    for folder in tqdm(os.listdir(root_dir)):
        action_path = os.path.join(root_dir, folder)
        if os.path.isdir(action_path):
            for video in os.listdir(os.path.join(root_dir, folder)):
                video_path = os.path.join(root_dir, folder, video)
                if os.path.exists(video_path.split(".")[0]):
                    continue
                if ".mp4" in video_path:
                    get_frames(video_path)
                    
        print(f"Generated Output for Action: {folder}")

if __name__ == '__main__':
    root_path = "/scratch/saali/Datasets/Multi_Moments_in_Time/videos"
    process_mmt(root_path)
