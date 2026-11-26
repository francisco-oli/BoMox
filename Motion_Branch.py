
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


RECON_STD_FLOOR = 0.01  # rad / m


def observed_channel_stats(smpl_batch, masks, num_channels=178, std_floor=RECON_STD_FLOOR):
  """Per-sample, per-channel mean / std over the observed frames. Returns mean, std of shape [B, 1, C]."""
  x = smpl_batch.squeeze(1).float()
  m = masks.to(x.device).float()
  n = m.sum(dim=-1).clamp(min=1.0)
  mean = (x * m).sum(dim=-1) / n
  var = (((x - mean.unsqueeze(-1)) ** 2) * m).sum(dim=-1) / n
  std = var.sqrt().clamp(min=std_floor)
  return mean.view(-1, 1, num_channels), std.view(-1, 1, num_channels)



class Motion_Branch(nn.Module):
  def __init__(self, motion_encoder, reconstructor, normalized_output=False):
    super().__init__()

    self.motion_encoder = motion_encoder
    self.reconstructor = reconstructor

    self.normalized_output = normalized_output

  def forward(self, x):

    final_emb, unpooled_seq = self.motion_encoder(x)
    output = self.reconstructor(x, final_emb, unpooled_seq)

    if self.normalized_output:
      # De-normalise with observed-frame stats
      mean, std = observed_channel_stats(x[0], x[1])
      output = output.float() * std + mean

    return output, final_emb


