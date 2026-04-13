import torch

# Configuration
base_ckpt_path = "/root/autodl-tmp/RFdiffusion/models/Base_ckpt.pt"
lora_ckpt_path = "lora_best.pt"
output_ckpt_path = "merged_base_ckpt_finetuned.pt"

LORA_R = 16
LORA_ALPHA = 32.0
scaling = LORA_ALPHA / LORA_R

base_ckpt = torch.load(base_ckpt_path, map_location="cpu")
base_state = base_ckpt['model_state_dict']

lora_state = torch.load(lora_ckpt_path, map_location="cpu")

merged_state = base_state.copy()

# Merge LoRA weights into base state dict
for key in list(lora_state.keys()):
    if key.endswith(".lora_A"):
        # Extract layer prefix
        prefix = key[:-7] 
        
        lora_A = lora_state[prefix + ".lora_A"].to(torch.float32)
        lora_B = lora_state[prefix + ".lora_B"].to(torch.float32)
        
        base_weight_key = prefix + ".weight"
        
        if base_weight_key in merged_state:
            base_w = merged_state[base_weight_key].to(torch.float32)
            
            # W_new = W_base + (B @ A) * scaling
            merged_state[base_weight_key] = base_w + (lora_B @ lora_A) * scaling

base_ckpt['model_state_dict'] = merged_state
torch.save(base_ckpt, output_ckpt_path)