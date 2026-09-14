
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


class Motion_Branch(nn.Module):
  def __init__(self, motion_encoder, reconstructor):
    super().__init__()

    self.motion_encoder = motion_encoder
    self.reconstructor = reconstructor

  def forward(self, x):

    final_emb, unpooled_seq = self.motion_encoder(x)
    output = self.reconstructor(x, final_emb, unpooled_seq)

    return output, final_emb
  

