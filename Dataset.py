
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
PATCH_LEN = 8  # MOMENT patch length

# SMPL-X channel groups
CHANNEL_GROUPS = {
    'trans': range(0, 3), 'root': range(3, 6), 'body': range(6, 69), 'jaw': range(69, 72),
    'eyes': range(72, 78), 'hands': range(78, 168), 'betas': range(168, 178),
}
# Channel groups to zero out, e.g. BOMOX_DROP_CHANNELS=hands,jaw,eyes,betas
DROP_CHANNELS = sorted(g for g in os.environ.get('BOMOX_DROP_CHANNELS', '').split(',') if g)
for _g in DROP_CHANNELS:
    if _g not in CHANNEL_GROUPS:
        raise ValueError(f'BOMOX_DROP_CHANNELS: unknown group {_g!r}; choose from {sorted(CHANNEL_GROUPS)}')
DROP_INDEX = sorted(i for g in DROP_CHANNELS for i in CHANNEL_GROUPS[g])
if DROP_CHANNELS:
    print(f'[Dataset] input channels removed (zeroed): {DROP_CHANNELS} ({len(DROP_INDEX)} of 178)')

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


def sample_masked_block(valid_length, masking_ratio, randint, patch_len=PATCH_LEN):
    """Patch-aligned contiguous masked block. Returns (start_idx, end_idx)."""
    block_size = max(1, int(valid_length * masking_ratio))

    if valid_length >= 2 * patch_len:
        block_size = max(patch_len, int(round(block_size / patch_len)) * patch_len)
        block_size = min(block_size, (valid_length // patch_len - 1) * patch_len)
        start_idx = int(randint(0, (valid_length - block_size) // patch_len + 1)) * patch_len
    else:
        block_size = min(block_size, valid_length)
        start_idx = int(randint(0, valid_length - block_size + 1))

    return start_idx, start_idx + block_size


class Multimodal_Dataset(Dataset):
    def __init__(self, motion_files, text_files, video_files, preprocess, num_frames,
                 window_size=512, mean_masking=0.1, train=True, apply_mask=True):

        self.motion_files = list(motion_files)
        self.text_files = list(text_files)
        self.video_files = list(video_files)

        stem = lambda p: os.path.splitext(os.path.basename(p))[0]
        bad = [i for i, (m, t, v) in enumerate(zip(self.motion_files, self.text_files, self.video_files))
               if not (stem(m) == stem(t) == stem(v))]
        if bad or not (len(self.motion_files) == len(self.text_files) == len(self.video_files)):
            raise ValueError(f"{len(bad)} misaligned (motion, text, video) triples, e.g. index {bad[:3]}")

        self.window_size = window_size
        self.n_frames = num_frames
        self.mean_masking = mean_masking
        self.train = train            # False: deterministic crop and mask
        self.apply_mask = apply_mask  # False: no masked block

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
            if DROP_INDEX:
                full_sequence[:, DROP_INDEX] = 0.0
            if len(full_sequence) < 8:
                raise ValueError(f"Sequence too short: {len(full_sequence)} frames")

            # train: random window and mask; eval: centred window, fixed mask
            rng = np.random if self.train else np.random.default_rng(idx)

            if self.train or len(full_sequence) <= self.window_size:
                window = get_random_window(full_sequence, self.window_size)
            else:
                start = (len(full_sequence) - self.window_size) // 2
                window = full_sequence[start : start + self.window_size].copy()

            # Padding mask
            mask = np.zeros(self.window_size, dtype=np.float32)
            valid_length = min(len(full_sequence), self.window_size)
            mask[:valid_length] = 1.0

            # Masked block for reconstruction
            if self.apply_mask:
                masking_ratio = rng.uniform(self.mean_masking - 0.05, self.mean_masking + 0.05)
                randint = rng.randint if self.train else rng.integers
                start_idx, end_idx = sample_masked_block(valid_length, masking_ratio, randint)
                mask[start_idx:end_idx] = 0.0

            video_file = torch.load(video_file_name)

            return torch.from_numpy(window).float(), torch.from_numpy(mask).float(), valid_length, text_file, video_file, motion_file_name

        except (json.JSONDecodeError, Exception) as e:
            if isinstance(e, ValueError) and "misaligned" in str(e):
                raise
            print(f"\n[WARNING] Skipping corrupt sample at index {idx}: {self.motion_files[idx]}. Error: {e}")
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)


def moment_collate_fn(batch):
    """Reshape the batch to MOMENT's [B*C, 1, L] format."""
    windows, masks, valid_length, text, video, motion_file_name = zip(*batch)
    windows = torch.stack(windows) # [B, L, C]
    masks = torch.stack(masks)     # [B, L]

    B, L, C = windows.shape

    # Prep for MOMENT: [B, L, C] -> [B, C, L] -> [B*C, 1, L]
    mx_batch = windows.permute(0, 2, 1).reshape(B * C, 1, L)

    # Expand masks to match channels: [B, L] -> [B*C, L]
    mx_masks = masks.repeat_interleave(C, dim=0)

    subsampled_videos = []

    for video_tensor in video:
        # video_tensor shape: [T_original, 3, 224, 224]
        T_orig = video_tensor.shape[0]

        if T_orig == 0:
            # Empty video fallback
            sampled = torch.zeros(NUM_FRAMES, 3, 224, 224)

        elif T_orig <= NUM_FRAMES:
            # Short video: repeat frames
            indices = torch.linspace(0, T_orig - 1, steps=NUM_FRAMES).long()
            sampled = video_tensor[indices]

        else:
            # Uniformly sample frames
            indices = torch.linspace(0, T_orig - 1, steps=NUM_FRAMES).long()
            sampled = video_tensor[indices]

        subsampled_videos.append(sampled)

    return mx_batch, mx_masks, list(valid_length), list(text), torch.stack(subsampled_videos), list(motion_file_name)