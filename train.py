import sys
import os
import json
import numpy as np
import argparse
import random
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas
from tqdm.auto import tqdm
from torch.amp import autocast, GradScaler

import Dataset 
import Model8.Motion_Encoder as Mencoder
import Model8.Motion_Reconstruction as Mreconstruct
import Model8.Motion_Branch as Motion
import Text_Encoder as Tencoder
import Video_Encoder as Vencoder
from data_split import build_split, unzip_triples


import wandb

LClip_internal_root = os.path.abspath("./Long-CLIP")
if LClip_internal_root not in sys.path:
    sys.path.insert(0, LClip_internal_root)

from model import longclip

#GLOBAL TRAINING VARIABLES

EPOCHS = 60
WEIGHT_DECAY = 1e-5
LEARNING_RATE = 1e-4
LOSS_BALANCE = 10
VALIDATE_EVERY_N_EPOCHS = 1
EARLY_STOPPING_PATIENCE = 15
CHECKPOINT_EVERY_N_EPOCHS = 10

# Global Variables
BATCH_SIZE = 32
NUM_FRAMES = 20
WINDOW_SIZE = 512
MHAP_INPUT_SHAPE = (NUM_FRAMES, 512)


CHECKPOINT_PATH = './Motion-X++/Motion-X++/checkpoints' 

STAMP_CONFIG = {
    'model_dim': 512,
    'n_layers': 4,
    'n_heads': 4,
    'dropout': 0.1,
}

MHAP_CONFIG = {
    'model_dim': 512,
    'n_layers': 4,
    'n_heads': 4,
    'dropout': 0.1,
}

def contrastive_loss(features_a, features_b, logit_scale):
    """
    Computes standard symmetric InfoNCE contrastive loss across a batch.
    """
    # Normalize features to unit vectors
    features_a = F.normalize(features_a, dim=-1)
    features_b = F.normalize(features_b, dim=-1)
    
    # Calculate similarity matrices
    scale = logit_scale.exp()
    logits_per_a = scale * torch.matmul(features_a, features_b.t())
    logits_per_b = logits_per_a.t()
    
    # Ground truth targets
    labels = torch.arange(features_a.size(0), device=features_a.device)
    
    loss_a = F.cross_entropy(logits_per_a, labels)
    loss_b = F.cross_entropy(logits_per_b, labels)
    
    return (loss_a + loss_b) / 2


def compute_batch_losses(batch, motion_branch, video_encoder, text_encoder, logit_scale, device):
    """
    Shared forward + loss computation, used IDENTICALLY by the training loop
    and the validation pass. Never duplicate this logic inline elsewhere --
    that's exactly how train.py and forward_pass.py's data splits drifted
    apart earlier in this project.

    Caller is responsible for the torch.no_grad() context (validation) or
    lack thereof (training), and for optimizer.zero_grad()/backward()/step()
    around this in the training case. This function only computes losses.

    Returns: total_loss, loss_recon, loss_contrastive, motion_output,
             motion_representation, video_output, text_output
    """
    motion_sample, motion_mask, valid_length, text_sample, video_sample, motion_file_name = batch

    valid_length = torch.tensor(valid_length, device=device)

    motion_sample = motion_sample.to(device)
    motion_mask = motion_mask.to(device)
    video_sample = video_sample.to(device)

    with autocast('cuda'):
        # Forward pass
        motion_output, motion_representation = motion_branch([motion_sample, motion_mask])
        text_output = text_encoder(text_sample)
        video_output = video_encoder(video_sample)

        current_batch_size = motion_output.size(0)
        target_motion_3d = motion_sample.view(current_batch_size, 178, WINDOW_SIZE)
        target_motion = target_motion_3d.permute(0, 2, 1)

        reshaped_masks = motion_mask.view(current_batch_size, 178, WINDOW_SIZE)
        single_masks = reshaped_masks[:, 0, :]

        range_tensor = torch.arange(WINDOW_SIZE, device=device).unsqueeze(0)
        len_mask = range_tensor < valid_length.unsqueeze(1)

        recon_mask = (single_masks == 0.0).float() * len_mask.float()
        recon_mask_3d = recon_mask.unsqueeze(-1)

        diff_sq = (motion_output - target_motion) ** 2
        masked_diff = diff_sq * recon_mask_3d

        # Normalize division by total unmasked items to avoid scale distortion
        num_masked_elements = recon_mask_3d.sum()
        if num_masked_elements > 0:
            loss_recon = masked_diff.sum() / (num_masked_elements * target_motion.size(-1))
        else:
            loss_recon = torch.tensor(0.0, device=device)

        # Cross-Modality Contrastive Loss
        loss_contrastive_video = contrastive_loss(motion_representation, video_output, logit_scale)
        loss_contrastive_text = contrastive_loss(motion_representation, text_output, logit_scale)
        loss_contrastive_mixed = contrastive_loss(video_output, text_output, logit_scale)
        loss_contrastive = (loss_contrastive_video + loss_contrastive_text + loss_contrastive_mixed) / 3.0

        # Combined Loss
        total_loss = loss_contrastive + (LOSS_BALANCE * loss_recon)

    return total_loss, loss_recon, loss_contrastive, motion_output, motion_representation, video_output, text_output


def run_validation(val_loader, motion_branch, video_encoder, text_encoder, logit_scale, device):
    """
    Runs a full pass over the validation set in eval mode.
    Returns loss metrics AND retrieval metrics (Top-1, Top-5, Top-10).
    """
    motion_branch.eval()
    video_encoder.eval()
    text_encoder.eval()

    total_sum, recon_sum, contrastive_sum = 0.0, 0.0, 0.0
    valid_count, nan_count = 0, 0

    # Collectors for retrieval metrics
    all_motion = []
    all_video = []
    all_text = []

    try:
        with torch.no_grad():
            for batch in val_loader:
                total_loss, loss_recon, loss_contrastive, motion_out, motion_rep, video_out, text_out = \
                    compute_batch_losses(batch, motion_branch, video_encoder, text_encoder, logit_scale, device)
                
                if not torch.isnan(total_loss):
                    total_sum += total_loss.item()
                    recon_sum += loss_recon.item()
                    contrastive_sum += loss_contrastive.item()
                    valid_count += 1
                    
                    # Store normalized embeddings for retrieval metrics
                    all_motion.append(F.normalize(motion_rep, dim=-1))
                    all_video.append(F.normalize(video_out, dim=-1))
                    all_text.append(F.normalize(text_out, dim=-1))
                else:
                    nan_count += 1
    finally:
        # Always restore train mode, even if validation raised partway through.
        motion_branch.train()
        video_encoder.train()
        text_encoder.train()

    # Compute averages for losses
    if valid_count > 0:
        avg_total = total_sum / valid_count
        avg_recon = recon_sum / valid_count
        avg_contrastive = contrastive_sum / valid_count
    else:
        return (float('nan'), float('nan'), float('nan'),
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, nan_count)

    # ----- COMPUTE RETRIEVAL METRICS -----
    # Concatenate all embeddings
    motion_emb = torch.cat(all_motion, dim=0).float()  # (N, 512)
    video_emb = torch.cat(all_video, dim=0).float()    # (N, 512)
    text_emb = torch.cat(all_text, dim=0).float()      # (N, 512)

    # Similarity matrices (Motion -> Video, Motion -> Text)
    sim_m2v = motion_emb @ video_emb.T
    sim_m2t = motion_emb @ text_emb.T

    # Ground truth indices (diagonal)
    labels = torch.arange(motion_emb.size(0), device=device)

    # Compute Top-1, Top-5, and Top-10 accuracy for Motion->Video
    m2v_top1 = (sim_m2v.topk(1, dim=1).indices == labels.unsqueeze(1)).float().mean().item()
    m2v_top5 = (sim_m2v.topk(5, dim=1).indices == labels.unsqueeze(1)).any(dim=1).float().mean().item()
    m2v_top10 = (sim_m2v.topk(10, dim=1).indices == labels.unsqueeze(1)).any(dim=1).float().mean().item()

    # Compute Top-1, Top-5, and Top-10 accuracy for Motion->Text
    m2t_top1 = (sim_m2t.topk(1, dim=1).indices == labels.unsqueeze(1)).float().mean().item()
    m2t_top5 = (sim_m2t.topk(5, dim=1).indices == labels.unsqueeze(1)).any(dim=1).float().mean().item()
    m2t_top10 = (sim_m2t.topk(10, dim=1).indices == labels.unsqueeze(1)).any(dim=1).float().mean().item()

    return (avg_total, avg_recon, avg_contrastive,
            m2v_top1, m2v_top5, m2v_top10,
            m2t_top1, m2t_top5, m2t_top10,
            nan_count)


def main():

    parser = argparse.ArgumentParser(description="Train Multimodal Motion Model")
    parser.add_argument(
        '--resume', 
        action='store_true', 
        help="Resume training from the latest checkpoint if it exists. Default is False (restart)."
    )
    parser.add_argument(
        '--params', 
        action='store_true', 
        help="Print trainable model parameters. Default is False "
    )

    parser.add_argument('--wandb_id', type=str, default=None, help="W&B Run ID to resume an existing dashboard.")

    args = parser.parse_args()

    print("Resume training? Answer:", args.resume)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Fetch Data -- via the shared split module, so this is guaranteed
    # identical to whatever forward_pass.py / eval scripts use.
    random.seed(42)

    split = build_split()
    train_mx, train_text, train_video = unzip_triples(split['train'])
    val_mx, val_text, val_video = unzip_triples(split['val'])
    test_mx, test_text, test_video = unzip_triples(split['test'])


    # Load Model Foundations
    long_clip_video, preprocess = longclip.load("./Long-CLIP/checkpoints/longclip-B.pt", device)
    long_clip_text, _ = longclip.load("./Long-CLIP/checkpoints/longclip-B.pt", device)

    
    # Freeze the backbone weights of Long-CLIP
    for param in long_clip_video.parameters():
        param.requires_grad = False

    for param in long_clip_text.parameters():
        param.requires_grad = False
    

    dataset = Dataset.Multimodal_Dataset(train_mx, train_text, train_video, preprocess, NUM_FRAMES)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=Dataset.moment_collate_fn,
        num_workers=8,            
        pin_memory=True,          
        persistent_workers=True   
    )

    val_dataset = Dataset.Multimodal_Dataset(val_mx, val_text, val_video, preprocess, NUM_FRAMES)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,  # no need to shuffle -- not used for gradient updates
        collate_fn=Dataset.moment_collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # Initialize Modules
    motion_encoder = Mencoder.Motion_Encoder(device, BATCH_SIZE, STAMP_CONFIG)
    reconstructor = Mreconstruct.Reconstruction_Transformer()
    text_encoder = Tencoder.Text_Encoder(long_clip_text, device)
    video_clip = Vencoder.CLIP_Video(long_clip_video, device)
    mhap = Vencoder.create_mhap(MHAP_INPUT_SHAPE, MHAP_CONFIG)

    motion_branch = Motion.Motion_Branch(motion_encoder, reconstructor).to(device)
    video_encoder = Vencoder.Video_Encoder(video_clip, mhap).to(device)
    text_encoder.to(device)

    def print_peft_trainable(model, name):
        print(f"\n--- Trainable params in {name} ---")
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                print(f"  {param_name}: {param.numel()} params")
        print(f"Total: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    # After instantiating text_encoder and video_encoder:
    #print_peft_trainable(text_encoder.text_encoder.transformer, "Text Encoder")
    #print_peft_trainable(video_encoder.clip_encoder.model, "Video Encoder")

    motion_branch = torch.compile(motion_branch)
    video_encoder = torch.compile(video_encoder)

    # Set modules to Training Mode
    motion_branch.train()
    video_encoder.train()
    text_encoder.train()

    # Learnable temperature
    logit_scale = nn.Parameter(torch.ones([], device=device) * np.log(1 / 0.07))

    # Gather Trainable Parameters & Optimizer
    trainable_params = (
        list(motion_branch.parameters()) + 
        list(video_encoder.parameters()) + 
        list(text_encoder.parameters()) + 
        [logit_scale]
    )

    # Sanity check trainable parameters
    if args.params:
        
        print("--- TRAINABLE PARAMETERS ---")
        total_params = 0
        for name, param in motion_branch.named_parameters():
            if param.requires_grad:
                print(f"Motion Branch -> {name}: {param.numel()}")
                total_params += param.numel()
        for name, param in video_encoder.named_parameters():
            if param.requires_grad:
                print(f"Video Encoder -> {name}: {param.numel()}")
                total_params += param.numel()
        for name, param in text_encoder.named_parameters():
            if param.requires_grad:
                print(f"Text Encoder  -> {name}: {param.numel()}")
                total_params += param.numel()
        if logit_scale.requires_grad:
            print(f"Logit Scale   -> {logit_scale.numel()}")
            total_params += logit_scale.numel()
        print(f"Total Trainable Parameters: {total_params:,}\n")

    
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    ) 
    
    scaler = GradScaler('cuda')
    epochs = EPOCHS

    # Scheduler now maximizes validation Motion→Video Top-1 accuracy
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)

    start_epoch = 0
    checkpoint_file = os.path.join(CHECKPOINT_PATH, 'model8_epoch20.pt')

    # Early stopping tracks best validation M2V Top-1 (higher is better)
    best_val_metric = 0.0
    bad_epochs = 0

    if args.resume:
        if os.path.exists(checkpoint_file):
            print(f"==> Explicitly resuming training from checkpoint: {checkpoint_file}")
            checkpoint = torch.load(checkpoint_file, map_location=device)
            
            # Load weights
            motion_branch.load_state_dict(checkpoint['motion_branch'])
            video_encoder.load_state_dict(checkpoint['video_encoder'])
            text_encoder.load_state_dict(checkpoint['text_encoder'])
            
            # Load scalar parameters 
            with torch.no_grad():
                logit_scale.copy_(checkpoint['logit_scale'])
                
            # Load optimizer and scaler states
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

            # ----- RESTORE LEARNING RATE -----
            if 'current_lr' in checkpoint:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = checkpoint['current_lr']
                print(f"==> Restored learning rate to {checkpoint['current_lr']:.2e}")
            else:
                print("==> Warning: No current_lr in checkpoint. Using default LR.")

            # ----- RESTORE SCHEDULER -----
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("==> Loaded scheduler state.")
            else:
                print("==> Warning: No scheduler state found. Starting fresh.")

            # ----- RESTORE EARLY STOPPING STATE -----
            # Try new key first, fallback to old key for compatibility
            if 'best_val_metric' in checkpoint:
                best_val_metric = checkpoint['best_val_metric']
                bad_epochs = checkpoint.get('bad_epochs', 0)
                print(f"==> Loaded early stopping state: best_val_metric={best_val_metric:.4f}, bad_epochs={bad_epochs}")
            elif 'best_val_recon' in checkpoint:
                # Legacy: best_val_recon stored reconstruction loss, but we now use metric
                # Convert to 0.0 (safe fallback) and reset bad_epochs
                best_val_metric = 0.0
                bad_epochs = 0
                print("==> Warning: Legacy checkpoint (best_val_recon) found. Resetting early stopping state.")
            else:
                print("==> No early stopping state found. Starting fresh.")
                
            # Set start epoch
            start_epoch = checkpoint['epoch'] + 1
            print(f"==> Successfully loaded checkpoint. Resuming from Epoch {start_epoch + 1}")
        else:
            print(f"==> Warning: '--resume' was specified, but no checkpoint was found at {checkpoint_file}.")
            print("==> Starting training from scratch instead.")
    else:
        print("==> Flag '--resume' not specified. Starting a fresh training run from scratch.")


 
    # Initialize W&B
    wandb.init(
        entity="francisco-oli-instituto-superior-t-cnico",
        project="BOMOX",
        id=args.wandb_id,      # Use the specific ID to resume
        resume="allow",        # Resumes if the ID exists, or creates a new run if it doesn't
        config={
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "loss_balance": LOSS_BALANCE,
            "num_frames": NUM_FRAMES,
            "window_size": WINDOW_SIZE
        }
    )
    
    # Optional: Track the global step for seamless charts when resuming
    global_step = start_epoch * len(loader)

    print("--- STARTING TRAINING LOOP ---")
    epoch = start_epoch - 1   # guard for skip (if start_epoch==0, epoch=-1, but loop runs at least once)
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        recon_loss_accum = 0.0
        contrastive_loss_accum = 0.0
        valid_batch_count = 0
        nan_batch_count = 0
        
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            optimizer.zero_grad()

            total_loss, loss_recon, loss_contrastive, motion_output, motion_representation, video_output, text_output = \
                compute_batch_losses(batch, motion_branch, video_encoder, text_encoder, logit_scale, device)


            # Backward pass 
            scaler.scale(total_loss).backward()


            scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            # ===== END GRADIENT CLIPPING =====

            scaler.step(optimizer)
            scaler.update()


            with torch.no_grad():
                logit_scale.clamp_(0, np.log(100))
            # ===== END LOGIT SCALE CLAMP =====

            # Track statistics
            if not torch.isnan(total_loss):
                epoch_loss += total_loss.item()
                recon_loss_accum += loss_recon.item()
                contrastive_loss_accum += loss_contrastive.item()
                valid_batch_count += 1
            else:
                nan_batch_count += 1
                print(f"\n[NaN WARNING] Epoch {epoch+1}, batch {progress_bar.n}: "
                      f"total_loss is NaN. Skipped (weights untouched this step). "
                      f"Running NaN count this epoch: {nan_batch_count}")
            
            wandb.log({
                    "train/total_loss": total_loss.item(),
                    "train/recon_loss": loss_recon.item(),
                    "train/contrastive_loss": loss_contrastive.item(),
                    "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "train/logit_scale": logit_scale.item(),
                    "epoch": epoch + 1 
                }, step=global_step)
            
            global_step += 1

            # ===== LIVE DIVERSITY CHECK (first batch of each epoch) =====
            if progress_bar.n == 0:
                with torch.no_grad():
                    rep = motion_representation.float()
                    rep_n = F.normalize(rep, dim=-1)
                    rep_cos = rep_n @ rep_n.t()
                    b = rep_cos.size(0)
                    rep_div = (rep_cos.sum() - b) / (b * b - b)

                    out = motion_output.float().reshape(motion_output.size(0), -1)
                    out_n = F.normalize(out, dim=-1)
                    out_cos = out_n @ out_n.t()
                    out_div = (out_cos.sum() - b) / (b * b - b)

                    print(f"[DIVERSITY] epoch {epoch+1} | "
                          f"motion_representation mean cos-sim: {rep_div.item():.4f} | "
                          f"motion_output mean cos-sim: {out_div.item():.4f} "
                          f"(lower = more varied across the batch; near 1.0 = collapsed)")
                    
                    wandb.log({
                        "diversity/rep_div": rep_div.item(),
                        "diversity/out_div": out_div.item(),
                    }, step=global_step)
            # ===== END LIVE DIVERSITY CHECK =====

    


        if valid_batch_count > 0:
            avg_loss = epoch_loss / valid_batch_count
            avg_recon = recon_loss_accum / valid_batch_count
            avg_contrastive = contrastive_loss_accum / valid_batch_count
        else:
            avg_loss = float('nan')
            avg_recon = float('nan')
            avg_contrastive = float('nan')

        print(f"-> Epoch {epoch+1} Completed. Avg Loss: {avg_loss:.4f} "
              f"(Recon: {avg_recon:.4f}, Contrast: {avg_contrastive:.4f}) "
              f"| valid_batches={valid_batch_count}, nan_batches={nan_batch_count}\n")

        if nan_batch_count > 0:
            nan_frac = nan_batch_count / (valid_batch_count + nan_batch_count)
            print(f"[NaN SUMMARY] Epoch {epoch+1}: {nan_batch_count} NaN batches "
                  f"({nan_frac*100:.1f}% of epoch).")
            if nan_frac > 0.5:
                print(f"[NaN SUMMARY] WARNING: over half this epoch's batches were NaN. "
                      f"The averages above are computed from a small, possibly "
                      f"unrepresentative subset of batches -- treat them with caution.")

        wandb.log({
            "epoch/avg_total_loss": avg_loss,
            "epoch/avg_recon_loss": avg_recon,
            "epoch/avg_contrastive_loss": avg_contrastive,
            "epoch/nan_batch_count": nan_batch_count,
            "epoch/valid_batch_count": valid_batch_count,
        }, step=global_step)

        # ===== VALIDATION =====
        if (epoch + 1) % VALIDATE_EVERY_N_EPOCHS == 0:
            (val_total, val_recon, val_contrastive,
             val_m2v_top1, val_m2v_top5, val_m2v_top10,
             val_m2t_top1, val_m2t_top5, val_m2t_top10,
             val_nan_count) = run_validation(
                val_loader, motion_branch, video_encoder, text_encoder, logit_scale, device
            )
            
            print(f"[VALIDATION] Epoch {epoch+1}:")
            print(f"  Losses -> Total={val_total:.4f}, Recon={val_recon:.4f}, Contrast={val_contrastive:.4f}")
            print(f"  M2V -> Top-1: {val_m2v_top1*100:.2f}% | Top-5: {val_m2v_top5*100:.2f}% | Top-10: {val_m2v_top10*100:.2f}%")
            print(f"  M2T -> Top-1: {val_m2t_top1*100:.2f}% | Top-5: {val_m2t_top5*100:.2f}% | Top-10: {val_m2t_top10*100:.2f}%")
            print(f"  (nan_batches={val_nan_count})\n")

            # Step the scheduler based on Motion->Video Top-1 (MAXIMIZE)
            scheduler.step(val_m2v_top1)

            # Early stopping and best checkpoint based on M2V Top-1
            if not np.isnan(val_m2v_top1):
                if val_m2v_top1 > best_val_metric:
                    best_val_metric = val_m2v_top1
                    bad_epochs = 0

                    # Save best checkpoint
                    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
                    best_path = os.path.join(CHECKPOINT_PATH, 'model8_best_val.pt')
                    torch.save({
                        'epoch': epoch,
                        'motion_branch': motion_branch.state_dict(),
                        'video_encoder': video_encoder.state_dict(),
                        'text_encoder': text_encoder.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'logit_scale': logit_scale,
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_m2v_top1': val_m2v_top1,
                        'best_val_metric': best_val_metric,
                        'bad_epochs': bad_epochs,
                        'current_lr': optimizer.param_groups[0]['lr'],
                    }, best_path)
                    print(f"[VALIDATION] New best M2V Top-1 ({val_m2v_top1*100:.2f}%) -- saved.")
                else:
                    bad_epochs += 1

            # Log everything to W&B
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({
                "val/avg_total_loss": val_total,
                "val/avg_recon_loss": val_recon,
                "val/avg_contrastive_loss": val_contrastive,
                "val/m2v_top1": val_m2v_top1,
                "val/m2v_top5": val_m2v_top5,
                "val/m2v_top10": val_m2v_top10,
                "val/m2t_top1": val_m2t_top1,
                "val/m2t_top5": val_m2t_top5,
                "val/m2t_top10": val_m2t_top10,
                "val/nan_batch_count": val_nan_count,
                "train/learning_rate": current_lr,
                "early_stopping/bad_epochs": bad_epochs,
            }, step=global_step)
        # ===== END VALIDATION =====

        # ===== Periodic checkpoint =====
        if (epoch + 1) % CHECKPOINT_EVERY_N_EPOCHS == 0:
            os.makedirs(CHECKPOINT_PATH, exist_ok=True)
            periodic_path = os.path.join(CHECKPOINT_PATH, f'model8_epoch{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'motion_branch': motion_branch.state_dict(),
                'video_encoder': video_encoder.state_dict(),
                'text_encoder': text_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'logit_scale': logit_scale,
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_metric': best_val_metric,
                'bad_epochs': bad_epochs,
                'current_lr': optimizer.param_groups[0]['lr'],
            }, periodic_path)
            print(f"[CHECKPOINT] Saved periodic checkpoint to {periodic_path}\n")


        # Early stopping check
        if bad_epochs >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs (no improvement for {EARLY_STOPPING_PATIENCE} epochs).")
            break


    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    model_dir = os.path.join(CHECKPOINT_PATH, 'model8.pt')

    # Save final checkpoint
    torch.save({
        'epoch': epoch,  
        'motion_branch': motion_branch.state_dict(),
        'video_encoder': video_encoder.state_dict(),
        'text_encoder': text_encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'logit_scale': logit_scale,
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_metric': best_val_metric,
        'bad_epochs': bad_epochs,
        'current_lr': optimizer.param_groups[0]['lr'], 
    }, model_dir)
 
    print(f"Model saved successfully to {model_dir}.") 

    wandb.finish()

if __name__ == "__main__":
    main()