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
from torch.amp import autocast

from momentfm import MOMENTPipeline

#STAMP Imports
stamp_internal_root = os.path.abspath("./STAMP")

if stamp_internal_root not in sys.path:
    sys.path.insert(0, stamp_internal_root)

from stamp.modeling.stamp import STAMP

class ModalityAwareSTAMP(STAMP):
      def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Modality Encodings
        # 0: Trans (3), 1: Poses (165), 2: Betas (10)
        self.modality_embedding = nn.Embedding(3, self.D)
        nn.init.normal_(self.modality_embedding.weight, std=0.02)

        #Match Index to correct embedding
        modality_indices = torch.zeros(self.n_spatial_channels, dtype=torch.long)

        modality_indices[0:3] = 0
        modality_indices[3:168] = 1
        modality_indices[168:] = 2

        self.register_buffer('modality_indices', modality_indices)

      def forward(self, x, return_attention=False):

        x = x.unsqueeze(2).float() # (Batch, 178 joints, 1, 1024)


        if hasattr(self, 'linear'):
            x = self.linear(x) #512 dim

        # Modality embeddings
        m = self.modality_embedding(self.modality_indices)

        # Reshape to broadcast
        x = x + m.view(1, self.n_spatial_channels, 1, self.D)

        # Positional embeddings
        if self.use_positional_embeddings:
            x = self.add_positional_embeddings(x)

        # Flatten for transformer
        B, S, T, D = x.shape
        x = x.view(B, S * T, D)

        # transformer
        if self.transformer_params['type'] == 'basic':
            x = self.transformer(x)
        elif self.transformer_params['type'] == 'criss_cross':
            x = self.transformer(x)

        if self.encoder_aggregation == 'mean_across_tokens':
            x = x.mean(dim=1)

        else:
          x, weights = self.multi_head_attention_pooling(x, B)

        return x


def create_stamp_model(feature_shape, config):
    """Initialize CUSTOM ModalityAwareSTAMP model."""

    batch_size, n_spatial, n_temporal, n_dim = feature_shape

    model = ModalityAwareSTAMP(
        input_dim=n_dim,
        D=config['model_dim'],
        n_classes=1, #Dummy value
        n_temporal_channels=n_temporal,
        n_spatial_channels=n_spatial,
        encoder_aggregation='attention_pooling',
        use_batch_norm=True,
        use_instance_norm=False,

        initial_proj_params={'type': 'full', 'dropout_rate': config['dropout']},

        transformer_params={
            'type': 'criss_cross',
            'n_layers': config['n_layers'],
            'n_heads': config['n_heads'],
            'dim_feedforward': 256,
            'dropout_rate': config['dropout'],
            'norm_first': True,
            'use_final_norm': True
        },

        final_classifier_params={
            'hidden_sizes': [64],
            'dropout_rate': config['dropout']
        },

        mhap_params = {
            'A': 4,
            'n_queries_per_head': 8,
            'dropout_rate': 0.3,
            'query_combination': 'weighted_sum',
            'lambda_for_residual': 0.5
        }
    )
    return model

class Motion_Encoder (nn.Module):
  def __init__(self, device, batch_size, stamp_config):
        super().__init__()

        self.device = device
        self.batch_size = batch_size

        moment_model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={"task_name": "embedding"},
        )
        moment_model.init()

        self.feature_extractor = moment_model

        # ===== RevIN eps fix (PERMANENT -- do not remove) =====
        # RevIN normalizes as (x - mean) / (stdev + eps). MOMENT's library
        # default eps=1e-5 is too small: a channel with near-zero real
        # variance in its visible frames (a joint that barely moved) makes
        # this division blow up to values in the thousands, which overflows
        # under autocast's fp16 math a few layers into the frozen encoder.
        # This caused a full NaN training run once already -- it must stay
        # here, in the class itself, not as an external monkeypatch in
        # train.py, so it can't be silently dropped in a future refactor.
        self.feature_extractor.normalizer.eps = 0.1

        _orig_normalize = self.feature_extractor.normalizer._normalize
        def _safe_normalize(x):
            out = _orig_normalize(x)
            # Hard backstop in case some other pathway still produces an
            # extreme value even with the raised eps.
            return torch.clamp(out, -20.0, 20.0)
        self.feature_extractor.normalizer._normalize = _safe_normalize
        # ===== END RevIN eps fix =====

        self.stamp_model = create_stamp_model((batch_size, 178, 1, 1024), stamp_config)


  def forward(self, x):

    smpl_batch, masks = x

    smpl_batch = smpl_batch.to(self.device)
    masks = masks.to(self.device)

    # Guard Rail --------------------------------

    patch_len = getattr(self.feature_extractor, 'patch_len', 8)
    
    # Get the actual shape of the masks tensor (which is B*C, seq_len)
    num_rows, seq_len = masks.shape
    
    # Reshape the mask to group frames into patches
    mask_patched = masks.view(num_rows, -1, patch_len)
    
    # A patch is only valid if ALL frames in it are visible (sum equals patch_len)
    valid_patches = (mask_patched.sum(dim=-1) == patch_len).long()
    
    # Find any rows in the batch that have exactly zero valid patches
    bad_rows = (valid_patches.sum(dim=-1) == 0)
    
    if bad_rows.any():
        # Force the first patch (frames 0 to patch_len - 1) to be fully visible 
        # ONLY for the rows that mathematically failed.
        masks[bad_rows, :patch_len] = 1.0

    # --------------------------------------------

    with autocast('cuda'):

      #Contrastive Path
      output_pooled = self.feature_extractor(x_enc=smpl_batch, input_mask=masks)
      emb = output_pooled.embeddings.view(-1, 178, 1024)
      emb = emb.to(self.device)
      final_emb = self.stamp_model(emb)

      output_unpooled = self.feature_extractor(x_enc=smpl_batch, input_mask=masks, reduction='none')
      unpooled_features = output_unpooled.embeddings.squeeze(1) # Shape: (B*178, 64, 1024)
      unpooled_seq = unpooled_features.view(-1, 178, 64, 1024)

    return final_emb, unpooled_seq