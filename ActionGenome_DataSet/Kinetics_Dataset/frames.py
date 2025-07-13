import os
import subprocess

def extract_frames(video_path, folder_path):
    output_folder = os.path.basename(video_path).split('.')[0]
    output_folder = os.path.join(folder_path, output_folder)

    print(f"Extracting frames from {video_path} to {output_folder}")

    os.makedirs(output_folder, exist_ok=True)

    output_pattern = os.path.join(output_folder, 'frame_%04d.jpg')
    command = [
        "ffmpeg", "-i", video_path, "-vf", "fps=1", output_pattern
    ]

    subprocess.run(command)
    print(f"Extracted frames from {video_path} to {output_folder}")

def process_folders(data_folder):
    for folder in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, folder)
        if os.path.isdir(folder_path):
            for video in os.listdir(folder_path):
                video_path = os.path.join(folder_path, video)
                extract_frames(video_path, folder_path)

if __name__ == '__main__':
    data_folder = "/scratch/saali/Datasets/kinetics-400/train_256"
    process_folders(data_folder)