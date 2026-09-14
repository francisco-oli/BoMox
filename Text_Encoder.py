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

from peft import LoraConfig, get_peft_model   # <-- new import
from patched_attention import patch_attention_in_transformer

LClip_internal_root = os.path.abspath("./Long-CLIP")

if LClip_internal_root not in sys.path:
    sys.path.insert(0, LClip_internal_root)


from model import longclip

class Text_Encoder(nn.Module):
    def __init__(self, long_clip, device):
        super().__init__()

        self.device = device

        patch_attention_in_transformer(long_clip.transformer)

        config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "v_proj", "out_proj"],   # same as video
            lora_dropout=0.1,
            bias="none",
            modules_to_save=None
        )
        # Wrap the text transformer with LoRA
        long_clip.transformer = get_peft_model(long_clip.transformer, config)
        long_clip.transformer.print_trainable_parameters()

        self.text_encoder = long_clip

    def forward(self, text_files):

      tokens = longclip.tokenize(text_files).to(self.device)
      
      text_features = self.text_encoder.encode_text(tokens)

      return text_features 