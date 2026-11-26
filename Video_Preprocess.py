import sys
import av
import os
import json
import numpy as np
import random
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas
from tqdm.auto import tqdm
from torch.cuda.amp import autocast



LClip_internal_root = os.path.abspath("./Long-CLIP")
if LClip_internal_root not in sys.path:
    sys.path.insert(0, LClip_internal_root)

from model import longclip


SAVE_PATH = './Motion-X++/Motion-X++/preprocessed_videos/perform'
VIDEO_PATH =  './Motion-X++/Motion-X++/videos/perform'
RANDOM_LENGTH = 20

def get_samples(directory, extension):
    all_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                all_files.append(os.path.join(root, file))
    return all_files, len(all_files)


def _load_video_frames(video_path, preprocessor):
        try:

            video_name = os.path.basename(video_path)
            final_name = os.path.splitext(video_name)[0]

            save_path = os.path.join(SAVE_PATH,f'{final_name}.pt')

            container = av.open(video_path)
            stream = container.streams.video[0]
            
            total_frames = stream.frames
            
            frames = []
            frame_idx = 0
            
            for frame in container.decode(video=0):
                
                img = frame.to_image().convert("RGB")
                proc = preprocessor(img)
                frames.append(proc)
            
                frame_idx += 1
                
            container.close()


            os.makedirs(SAVE_PATH, exist_ok=True)

            if frames:
                video_tensor = torch.stack(frames)
            else:
                video_tensor = torch.zeros(0, 3, 224, 224)


            torch.save(video_tensor, save_path)
  

        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            video_tensor = torch.zeros(RANDOM_LENGTH, 3, 224, 224)

            torch.save(video_tensor, save_path)
            



def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    video_samples, _ = get_samples(VIDEO_PATH, '.mp4')

    print('Videos to process:', len(video_samples))

    long_clip, preprocess = longclip.load("./Long-CLIP/checkpoints/longclip-B.pt", device)

    for video in tqdm(video_samples, desc="Processing videos"):
        frame_list = _load_video_frames(video, preprocess)
        
    print('Finished preprocessing')
    



if __name__ == "__main__":
    main()
