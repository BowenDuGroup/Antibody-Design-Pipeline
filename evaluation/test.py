#!/usr/bin/env python
import os
import sys

# Set environment variables before framework import
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["OMP_NUM_THREADS"] = "4" 

RF_ROOT = "/root/autodl-tmp/RFdiffusion"
if RF_ROOT not in sys.path:
    sys.path.append(RF_ROOT)

import re
import time
import pickle
import logging
import random
import glob
import numpy as np
import torch
from omegaconf import OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
from rfdiffusion.util import writepdb_multi, writepdb
from rfdiffusion.inference import utils as iu

def make_deterministic(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

@hydra.main(version_base=None, config_path="../RFdiffusion/config/inference", config_name="base")
def main(conf: HydraConfig) -> None:
    log = logging.getLogger(__name__)

    CKPT_PATH = "/root/autodl-tmp/code/merged_rf_finetuned.pt"  
    OUTPUT_DIR = "/root/autodl-tmp/generation_results/test"
    MODEL_DIR = "/root/autodl-tmp/RFdiffusion/models"
    INPUT_PDB = "/root/autodl-tmp/code/6x1t.pdb" 
    CONTIG_SETTING = ["A2-100/0 50-60"]
    NUM_DESIGNS = 5
    
    # Override Hydra configs
    conf.inference.ckpt_override_path = CKPT_PATH
    conf.inference.output_prefix = OUTPUT_DIR
    conf.inference.model_directory_path = MODEL_DIR
    conf.inference.input_pdb = INPUT_PDB
    conf.inference.num_designs = NUM_DESIGNS
    conf.contigmap.contigs = CONTIG_SETTING

    make_deterministic()

    sampler = iu.sampler_selector(conf)
    sampler.model.eval()

    # Determine starting index to prevent overwriting
    design_startnum = sampler.inf_conf.design_startnum
    if sampler.inf_conf.design_startnum == -1:
        existing = glob.glob(sampler.inf_conf.output_prefix + "*.pdb")
        indices = [-1]
        for e in existing:
            m = re.match(r".*_(\d+)\.pdb$", e)
            if not m: continue
            indices.append(int(m.groups()[0]))
        design_startnum = max(indices) + 1

    # Inference loop
    for step, i_des in enumerate(range(design_startnum, design_startnum + NUM_DESIGNS), 1):
        start_time = time.time()
        out_prefix = f"{sampler.inf_conf.output_prefix}_{i_des}"
        
        x_init, seq_init = sampler.sample_init()
        denoised_xyz_stack = []
        px0_xyz_stack = []
        seq_stack = []
        plddt_stack = []

        x_t = torch.clone(x_init)
        seq_t = torch.clone(seq_init)
        
        # Denoising steps
        for t in range(int(sampler.t_step_input), sampler.inf_conf.final_step - 1, -1):
            px0, x_t, seq_t, plddt = sampler.sample_step(
                t=t, x_t=x_t, seq_init=seq_t, final_step=sampler.inf_conf.final_step
            )
            px0_xyz_stack.append(px0)
            denoised_xyz_stack.append(x_t)
            seq_stack.append(seq_t)
            plddt_stack.append(plddt[0])

        denoised_xyz_stack = torch.flip(torch.stack(denoised_xyz_stack), [0,])
        px0_xyz_stack = torch.flip(torch.stack(px0_xyz_stack), [0,])
        plddt_stack = torch.stack(plddt_stack)

        os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
        final_seq = seq_stack[-1]

        # Process placeholder residues
        final_seq = torch.where(torch.argmax(seq_init, dim=-1) == 21, 7, torch.argmax(seq_init, dim=-1))
        bfacts = torch.ones_like(final_seq.squeeze())
        bfacts[torch.where(torch.argmax(seq_init, dim=-1) == 21, True, False)] = 0
        
        out_pdb = f"{out_prefix}.pdb"
        writepdb(
            out_pdb,
            denoised_xyz_stack[0, :, :4],
            final_seq,
            sampler.binderlen,
            chain_idx=sampler.chain_idx,
            bfacts=bfacts,
            idx_pdb=sampler.idx_pdb
        )

        # Save metadata
        trb = dict(
            config=OmegaConf.to_container(sampler._conf, resolve=True),
            plddt=plddt_stack.cpu().numpy(),
            time=time.time() - start_time,
        )
        if hasattr(sampler, "contig_map"):
            for key, value in sampler.contig_map.get_mappings().items():
                trb[key] = value
                
        with open(f"{out_prefix}.trb", "wb") as f_out:
            pickle.dump(trb, f_out)

        # Save trajectory
        if sampler.inf_conf.write_trajectory:
            traj_prefix = os.path.dirname(out_prefix) + "/traj/" + os.path.basename(out_prefix)
            os.makedirs(os.path.dirname(traj_prefix), exist_ok=True)
            writepdb_multi(
                f"{traj_prefix}_Xt-1_traj.pdb",
                denoised_xyz_stack,
                bfacts,
                final_seq.squeeze(),
                use_hydrogens=False,
                backbone_only=False,
                chain_ids=sampler.chain_idx
            )

if __name__ == "__main__":
    main()