#Package imports
import sys
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
import av

from transformers import CLIPProcessor
from peft import LoraConfig, get_peft_model
from transformers import CLIPVisionModel

from patched_attention import patch_attention_in_transformer

#STAMP Imports
stamp_internal_root = os.path.abspath("./STAMP")

if stamp_internal_root not in sys.path:
    sys.path.insert(0, stamp_internal_root)

from stamp.modeling.stamp import STAMP



class CLIP_Video(nn.Module):
  def __init__(self, model, device):
    super().__init__()

    video_encoder = model

    # Replace nn.MultiheadAttention with an explicit-projection equivalent
    # BEFORE LoRA is attached, so q_proj/v_proj/out_proj are real submodules
    # that actually get called during forward() -- see PatchedMultiheadAttention
    # docstring above for why this is necessary.
    patch_attention_in_transformer(video_encoder.visual.transformer)

    config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "v_proj", "out_proj"],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=None
        )

    self.model = get_peft_model(video_encoder, config)
    self.model.print_trainable_parameters()

    self.device = device

    #self.linear = nn.Linear(video_encoder.config.hidden_size, 512)

  def forward(self, x):

    batch_size, num_frames, c, h, w = x.shape
    x_flattened = x.reshape(-1, c, h, w) #Reshape for CLIP
    x_flattened = x_flattened.to(self.device)

    emb = self.model.encode_image(x_flattened).view(batch_size, num_frames, -1)

    #emb = out.pooler_output.view(batch_size, num_frames, -1)
    #emb = self.linear(emb)

    return emb
  

class Video_MHAP(STAMP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, return_attention=False):

        B, T, D_in = x.shape

        x, weights = self.multi_head_attention_pooling(x, B)

        return x
      


def create_mhap(feature_shape, config):

    n_temporal, n_dim = feature_shape

    model = Video_MHAP(
        input_dim=n_dim,
        D=config['model_dim'],
        n_classes=1, #Dummy value
        n_temporal_channels=n_temporal,
        n_spatial_channels=1,
        encoder_aggregation='attention_pooling',
        use_batch_norm=True,
        use_instance_norm=False,

        mhap_params = {
            'A': 4,
            'n_queries_per_head': 8,
            'dropout_rate': 0.3,
            'query_combination': 'weighted_sum',
            'lambda_for_residual': 0.5
        },

        #Dummy
        final_classifier_params={
            'hidden_sizes': [64],
            'dropout_rate': config['dropout']
        },

        #Dummy
        initial_proj_params={'type': 'full', 'dropout_rate': config['dropout']},

    )
    return model



class Video_Encoder(nn.Module):
  def __init__(self, clip, mhap):
    super().__init__()

    self.clip_encoder = clip
    self.mhap = mhap

  def forward(self, x):

    clip_out = self.clip_encoder(x).to(torch.float32)
    mhap_out = self.mhap(clip_out)

    return mhap_out