
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



NUM_FRAMES = 20

def process_motion_x(data):

    frames = []
    body_shape = np.array(data['annotations'][0]['smplx_params']['betas']).flatten()[:10]

    for ann in data['annotations']:
        p = ann['smplx_params']

        trans = np.array(p['trans']).flatten().copy()
        root_pose = np.array(p['root_orient']).flatten()
        body_pose = np.array(p['pose_body']).flatten()
        jaw_pose = np.array(p['pose_jaw']).flatten()
        hand_pose = np.array(p['pose_hand']).flatten()
        eye_pose = np.zeros(6)

        pose_165 = np.concatenate([root_pose, body_pose, jaw_pose, eye_pose, hand_pose])
        full_frame = np.concatenate([trans, pose_165, body_shape])
        frames.append(full_frame)

    poses = np.array(frames)

    return poses


def get_random_window(pose_array, window_size):

    num_frames = pose_array.shape[0]
    feat_dim = pose_array.shape[1]

    if num_frames >= window_size:
        start = np.random.randint(0, num_frames - window_size + 1)
        window = pose_array[start : start + window_size].copy()
    else:
        pad_amount = window_size - num_frames
        padding = np.zeros((pad_amount, feat_dim))
        window = np.concatenate([pose_array, padding], axis=0)

    return window




class Multimodal_Dataset(Dataset):
    def __init__(self, motion_files, text_files, video_files, preprocess, num_frames, window_size=512, mean_masking = 0.1):

        self.motion_files = sorted(motion_files)
        self.text_files = sorted(text_files)
        self.video_files = sorted(video_files)

        self.window_size = window_size
        self.n_frames = num_frames
        self.mean_masking = mean_masking

    def __len__(self):
        return len(self.motion_files)

    def __getitem__(self, idx):

        try:
            motion_file_name = self.motion_files[idx]
            text_file_name = self.text_files[idx]
            video_file_name = self.video_files[idx]

            with open(text_file_name, 'r') as f:
                text_file = f.read()

            with open(motion_file_name, 'r') as f:
                raw_data = json.load(f)

            full_sequence = process_motion_x(raw_data)
            if len(full_sequence) < 8:
                raise ValueError(f"Sequence too short: {len(full_sequence)} frames")

            window = get_random_window(full_sequence, self.window_size)

            # padding mask: 1 for real frames, 0 for padded ones
            mask = np.zeros(self.window_size, dtype=np.float32)
            valid_length = min(len(full_sequence), self.window_size)
            mask[:valid_length] = 1.0

            # additionally mask out a random contiguous block for the reconstruction target
            masking_ratio = np.random.uniform(self.mean_masking - 0.05, self.mean_masking + 0.05)
            block_size = max(1, int(valid_length * masking_ratio))
            block_size = min(block_size, valid_length)

            max_start_idx = valid_length - block_size
            start_idx = np.random.randint(0, max_start_idx + 1)
            end_idx = start_idx + block_size
            mask[start_idx:end_idx] = 0.0

            video_file = torch.load(video_file_name)

            return torch.from_numpy(window).float(), torch.from_numpy(mask).float(), valid_length, text_file, video_file, motion_file_name

        except (json.JSONDecodeError, Exception) as e:
            print(f"\n[WARNING] Skipping corrupt sample at index {idx}: {self.motion_files[idx]}. Error: {e}")
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)
        
    
    
def moment_collate_fn(batch):
    """Reshapes the batch to MOMENT's expected [B*C, 1, L] layout."""
    windows, masks, valid_length, text, video, motion_file_name = zip(*batch)
    windows = torch.stack(windows)  # [B, L, C]
    masks = torch.stack(masks)      # [B, L]

    B, L, C = windows.shape

    # [B, L, C] -> [B, C, L] -> [B*C, 1, L]
    mx_batch = windows.permute(0, 2, 1).reshape(B * C, 1, L)
    mx_masks = masks.repeat_interleave(C, dim=0)  # [B, L] -> [B*C, L]

    subsampled_videos = []
    for video_tensor in video:
        T_orig = video_tensor.shape[0]

        if T_orig == 0:
            sampled = torch.zeros(NUM_FRAMES, 3, 224, 224)
        else:
            # linspace covers short and long videos alike, always including first/last frame
            indices = torch.linspace(0, T_orig - 1, steps=NUM_FRAMES).long()
            sampled = video_tensor[indices]

        subsampled_videos.append(sampled)

    return mx_batch, mx_masks, list(valid_length), list(text), torch.stack(subsampled_videos), list(motion_file_name)