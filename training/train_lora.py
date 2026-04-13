#!/usr/bin/env python
import torch
import numpy as np
import os
import sys
import time
import gc
import hydra
import math
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint as torch_checkpoint

# VRAM Management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# ==============================================================================
# Environment Patches
# ==============================================================================
def robust_from_numpy(x):
    x_arr = np.asarray(x)
    mapping = {np.int32: torch.int32, np.int64: torch.int64,
               np.float32: torch.float32, np.float64: torch.float64, bool: torch.bool}
    return torch.tensor(x_arr.tolist(), dtype=mapping.get(
        x_arr.dtype.type if hasattr(x_arr.dtype, 'type') else x_arr.dtype))
torch.from_numpy = robust_from_numpy

from dataset import AntibodyComplexDataset

RF_ROOT = "/root/autodl-tmp/RFdiffusion"
CKPT_PATH = "/root/autodl-tmp/RFdiffusion/models/Base_ckpt.pt"
PDB_DIR = "/root/autodl-tmp/BigBind"

if RF_ROOT not in sys.path:
    sys.path.append(RF_ROOT)
from rfdiffusion.RoseTTAFoldModel import RoseTTAFoldModule

# ==============================================================================
# Memory Diagnostics
# ==============================================================================
def print_gpu_memory(tag=""):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[{tag}] Allocated: {alloc:.2f}GB | Peak: {peak:.2f}GB | Total: {total:.1f}GB")

# ==============================================================================
# LoRA Implementation
# ==============================================================================
class LoRALayer(nn.Module):
    def __init__(self, original_linear: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original = original_linear
        in_dim = original_linear.in_features
        out_dim = original_linear.out_features

        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        weight_device = original_linear.weight.device
        self.lora_A = nn.Parameter(torch.zeros(in_dim, r, device=weight_device))
        self.lora_B = nn.Parameter(torch.zeros(r, out_dim, device=weight_device))
        self.scaling = alpha / r

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        orig_out = self.original(x)
        shape = x.shape
        x_2d = x.reshape(-1, shape[-1])
        lora_out = (x_2d @ self.lora_A) @ self.lora_B * self.scaling
        return orig_out + lora_out.reshape(*shape[:-1], -1)

def inject_lora(model, target_keywords, exclude_keywords=None, r=8, alpha=16.0):
    """
    Injects LoRA layers with support for keyword exclusion to avoid incompatible modules (e.g., SE3).
    """
    if exclude_keywords is None:
        exclude_keywords = []
    targets = []
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                full_name = f"{name}.{child_name}" if name else child_name
                
                if not any(kw in full_name.lower() for kw in target_keywords):
                    continue
                if any(ex in full_name.lower() for ex in exclude_keywords):
                    continue
                if child.in_features < 32 or child.out_features < 32:
                    continue
                targets.append((module, child_name, child, full_name))
                
    for parent, child_name, original_linear, _ in targets:
        setattr(parent, child_name, LoRALayer(original_linear, r=r, alpha=alpha))
    return len(targets)

def save_lora_weights(model, path):
    lora_state = {name: param.data.cpu()
                  for name, param in model.named_parameters() if param.requires_grad}
    torch.save(lora_state, path)
    return len(lora_state)

def load_lora_weights(model, path, device='cpu'):
    lora_state = torch.load(path, map_location=device)
    model_params = dict(model.named_parameters())
    loaded = 0
    for name, weight in lora_state.items():
        if name in model_params:
            model_params[name].data.copy_(weight)
            loaded += 1
    return loaded

# ==============================================================================
# Gradient Checkpointing via Monkey-patching
# ==============================================================================
def apply_gradient_checkpointing(model):
    """
    Wraps IterativeSimulator blocks with torch.utils.checkpoint to trade compute for VRAM.
    Reduces memory footprint by 60-70%.
    """
    patched = 0
    for name, module in model.named_modules():
        block_lists = []
        if hasattr(module, 'extra_block'):
            block_lists.append(('extra_block', module.extra_block))
        if hasattr(module, 'main_block'):
            block_lists.append(('main_block', module.main_block))

        for list_name, block_list in block_lists:
            for i in range(len(block_list)):
                block = block_list[i]
                original_forward = block.forward

                def make_wrapper(fn):
                    def wrapper(*args, **kwargs):
                        return torch_checkpoint(fn, *args, use_reentrant=False, **kwargs)
                    return wrapper

                block.forward = make_wrapper(original_forward)
                patched += 1

    return patched

# ==============================================================================
# Model Wrapper
# ==============================================================================
class RFDiffusionLoRAWrapper(nn.Module):
    def __init__(self, ckpt_path, device, lora_r=8, lora_alpha=16.0):
        super().__init__()
        self.device = device

        GlobalHydra.instance().clear()
        hydra.initialize_config_dir(
            config_dir=os.path.join(RF_ROOT, "config/inference"), version_base=None
        )
        conf = hydra.compose(
            config_name="base",
            overrides=[f"inference.ckpt_override_path={ckpt_path}"]
        )

        model_conf_dict = OmegaConf.to_container(conf.model, resolve=True)
        self.model = RoseTTAFoldModule(
            d_t1d=conf.preprocess.d_t1d, d_t2d=conf.preprocess.d_t2d,
            T=conf.diffuser.T, **model_conf_dict
        ).to(device)

        self.model.load_state_dict(
            torch.load(ckpt_path, map_location='cpu')['model_state_dict'],
            strict=False
        )

        n_patched = apply_gradient_checkpointing(self.model)
        print(f"[Init] Gradient Checkpointing activated for {n_patched} trunk blocks.")

        # Freeze all base parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Inject LoRA - Exclude non-standard tensor shapes in SE3
        lora_targets = ['str2str', 'ipa', 'pred', 'head']
        lora_excludes = ['se3', 'conv', 'radial', 'basis', 'fiber', 'norm']
        n_injected = inject_lora(self.model, lora_targets, 
                                 exclude_keywords=lora_excludes,
                                 r=lora_r, alpha=lora_alpha)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"[Init] LoRA injected: {n_injected} layers (r={lora_r}, alpha={lora_alpha})")
        print(f"[Init] Parameters: {total_params/1e6:.1f}M Total | {trainable_params/1e6:.2f}M Trainable ({100*trainable_params/total_params:.2f}%)")

        self.model.train()

    def forward(self, xyz_noisy, seq, t):
        B, L = xyz_noisy.shape[0], xyz_noisy.shape[1]
        seq_onehot = F.one_hot(seq, num_classes=22).float().to(self.device)

        args = {
            "msa_latent": torch.zeros((B, 1, L, 48), device=self.device),
            "msa_full":   torch.zeros((B, 1, L, 25), device=self.device),
            "seq":        seq_onehot,
            "idx":        torch.arange(L, device=self.device).unsqueeze(0).expand(B, -1),
            "t1d":        torch.zeros((B, 1, L, 22), device=self.device),
            "t2d":        torch.zeros((B, 1, L, L, 44), device=self.device),
            "xyz_t":      torch.zeros((B, 1, L, 27, 3), device=self.device),
            "alpha_t":    torch.zeros((B, 1, L, 30), device=self.device),
            "xyz":        torch.zeros((B, L, 27, 3), device=self.device).index_copy_(
                              2, torch.arange(3, device=self.device), xyz_noisy.float()
                          ),
            "t": t,
            "motif_mask": torch.zeros(L, dtype=torch.bool, device=self.device),
            "return_infer": True
        }

        outputs = self.model(**args)

        xyz_tensors = []
        def find_t(obj):
            if isinstance(obj, torch.Tensor) and obj.dim() == 4 and obj.shape[-1] == 3:
                xyz_tensors.append(obj)
            elif isinstance(obj, (tuple, list)):
                [find_t(i) for i in obj]
            elif isinstance(obj, dict):
                [find_t(v) for v in obj.values()]
        find_t(outputs)
        return xyz_tensors[-1][:, :, :3, :]

# ==============================================================================
# Main Training Loop
# ==============================================================================
def train_loop():
    device = torch.device('cuda')
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    dataset = AntibodyComplexDataset(PDB_DIR)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    print("\n[Dataset] Sequence Length Distribution:")
    lengths = []
    for i in range(len(dataset)):
        try:
            lengths.append(dataset[i]['xyz'].shape[0])
        except Exception:
            continue
    lengths.sort()
    
    if lengths:
        print(f"   Total pairs: {len(lengths)} | Min: {lengths[0]} | Max: {lengths[-1]} | Median: {lengths[len(lengths)//2]}")
        for thr in [400, 500, 600, 700, 800, 1000, 1500]:
            cnt = sum(1 for l in lengths if l <= thr)
            print(f"   L <= {thr}: {cnt} pairs ({100*cnt/len(lengths):.1f}%)")

    # Hyperparameters
    LORA_R = 16              
    LORA_ALPHA = 32.0        
    MAX_SEQ_LEN = 1500       
    TOTAL_EPOCHS = 80        
    GRAD_ACCUM_STEPS = 8     
    LR = 1e-4                
    WARMUP_EPOCHS = 5        
    DIFFUSION_T = 200        

    usable = sum(1 for l in lengths if l <= MAX_SEQ_LEN)
    print(f"\n[Config] MAX_SEQ_LEN={MAX_SEQ_LEN} -> Usable pairs: {usable}/{len(lengths)}")

    print_gpu_memory("Pre-Model Load")
    model = RFDiffusionLoRAWrapper(CKPT_PATH, device, lora_r=LORA_R, lora_alpha=LORA_ALPHA)
    print_gpu_memory("Post-Model Load")

    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=LR, weight_decay=0.01)

    # Warmup + Cosine LR Scheduler
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return 0.1 + 0.9 * epoch / WARMUP_EPOCHS
        else:
            progress = (epoch - WARMUP_EPOCHS) / max(TOTAL_EPOCHS - WARMUP_EPOCHS, 1)
            return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\n{'='*60}")
    print(f"Initiating RFdiffusion LoRA Fine-Tuning Pipeline")
    print(f"   LoRA r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"   LR: {LR} (Warmup {WARMUP_EPOCHS} ep -> Cosine -> {LR*0.02:.1e})")
    print(f"   Diffusion T={DIFFUSION_T} | Epochs: {TOTAL_EPOCHS}")
    print(f"   Gradient Accumulation Steps: {GRAD_ACCUM_STEPS} | Max Length: {MAX_SEQ_LEN}")
    print(f"{'='*60}\n")

    best_loss = float('inf')
    no_improve_count = 0

    for epoch in range(TOTAL_EPOCHS):
        epoch_loss, valid_steps, skipped_oom, skipped_len = 0.0, 0, 0, 0
        accum_count = 0
        start_time = time.time()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()

        for batch in dataloader:
            xyz_true = batch['xyz'].to(device).float()
            seq = batch['seq'].to(device)
            mask = batch['diffuse_mask'].to(device)
            seq_len = xyz_true.shape[1]

            if seq_len > MAX_SEQ_LEN:
                skipped_len += 1
                del xyz_true, seq, mask
                continue
            if mask.sum() == 0:
                del xyz_true, seq, mask
                continue

            try:
                # Cosine noise schedule mapping T=200
                t = torch.randint(1, DIFFUSION_T + 1, (1,)).to(device)
                s = 0.008
                alpha_bar = torch.cos(((t.float() / DIFFUSION_T + s) / (1 + s)) * math.pi / 2) ** 2
                sigma = torch.sqrt(1 - alpha_bar).clamp(min=0.01, max=1.0)

                xyz_noisy = xyz_true.clone()
                noise = torch.randn_like(xyz_true[mask]) * sigma
                xyz_noisy[mask] += noise

                pred = model(xyz_noisy, seq, t)
                loss = F.mse_loss(pred[mask], xyz_true[mask])

                loss_scaled = loss / GRAD_ACCUM_STEPS
                loss_scaled.backward()
                accum_count += 1

                if accum_count >= GRAD_ACCUM_STEPS:
                    torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    accum_count = 0

                epoch_loss += loss.item()
                valid_steps += 1

                if valid_steps <= 3 or valid_steps % 5 == 0:
                    mem_peak = torch.cuda.max_memory_allocated(device) / 1024**3
                    print(f"  [Epoch {epoch+1}] Step {valid_steps} | L={seq_len} | "
                          f"Loss: {loss.item():.4f} | VRAM Peak: {mem_peak:.1f}GB")

                del pred, loss, loss_scaled, xyz_noisy, noise
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    skipped_oom += 1
                    mem_peak = torch.cuda.max_memory_allocated(device) / 1024**3
                    print(f"  [OOM Warning] Skipped L={seq_len} (Peak VRAM: {mem_peak:.1f}GB)")
                    optimizer.zero_grad(set_to_none=True)
                    accum_count = 0
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e

            del xyz_true, seq, mask

        if accum_count > 0:
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        avg_loss = epoch_loss / valid_steps if valid_steps > 0 else 0
        current_lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{TOTAL_EPOCHS} Summary")
        print(f"   Average Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
        print(f"   Valid Steps: {valid_steps} | OOM Skipped: {skipped_oom} | Length Skipped: {skipped_len}")
        print(f"   Elapsed Time: {elapsed:.1f}s")
        print_gpu_memory(f"Epoch {epoch+1}")
        print(f"{'='*60}")

        if valid_steps > 0:
            if (epoch + 1) % 5 == 0:
                lora_path = f"lora_epoch_{epoch+1}.pt"
                n_saved = save_lora_weights(model.model, lora_path)
                fsize = os.path.getsize(lora_path) / 1024**2
                print(f"[Checkpoint] LoRA saved: {lora_path} ({n_saved} modules, {fsize:.1f}MB)")

            if avg_loss < best_loss:
                best_loss = avg_loss
                save_lora_weights(model.model, "lora_best.pt")
                no_improve_count = 0
                print(f"[Performance] New best model achieved (Loss: {best_loss:.4f})")
            else:
                no_improve_count += 1
                if no_improve_count >= 15:
                    print(f"[Early Stopping] No improvement for {no_improve_count} epochs. Terminating training.")
                    break

            if (epoch + 1) % 20 == 0:
                torch.save(model.model.state_dict(), f"full_model_epoch_{epoch+1}.pt")
                print(f"[Checkpoint] Full model saved: full_model_epoch_{epoch+1}.pt")
        else:
            print("[Warning] No valid steps completed in this epoch.")
        print()

if __name__ == "__main__":
    train_loop()