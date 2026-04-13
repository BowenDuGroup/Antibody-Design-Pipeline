#!/usr/bin/env python
import os
import sys
import shutil
import subprocess
import logging
import time

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] - %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# Configuration
MPNN_ROOT = "/root/autodl-tmp/ProteinMPNN"
TARGET_PDB = "/root/autodl-tmp/generation_results/test_1.pdb" 
OUTPUT_DIR = "/root/autodl-tmp/mpnn_results"

FIXED_CHAIN = "A"
DESIGN_CHAIN = "B"
NUM_SEQS = 8
TEMPERATURE = "0.1" 
BATCH_SIZE = 1
SEED = 42

def run_cmd(cmd, cwd=None):
    log.info(f"Executing: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        log.error(f"Execution failed:\n{result.stderr}")
        sys.exit(1)
    return result.stdout

def main():
    if not os.path.exists(MPNN_ROOT) or not os.path.exists(TARGET_PDB):
        log.error("Missing ProteinMPNN root directory or target PDB.")
        return

    # Setup temporary workspace
    temp_input_dir = os.path.join(OUTPUT_DIR, "temp_inputs")
    os.makedirs(temp_input_dir, exist_ok=True)
    
    pdb_name = os.path.basename(TARGET_PDB)
    work_pdb = os.path.join(temp_input_dir, pdb_name)
    shutil.copy(TARGET_PDB, work_pdb)
    
    json_parsed = os.path.join(temp_input_dir, "parsed_pdbs.jsonl")
    json_assigned = os.path.join(temp_input_dir, "assigned_chains.jsonl")

    # Step 1: Parse PDB coordinates
    run_cmd(f"python helper_scripts/parse_multiple_chains.py --input_path {temp_input_dir} --output_path {json_parsed}", cwd=MPNN_ROOT)

    # Step 2: Assign chains (Fix target, unlock design region)
    cmd2 = (f"python helper_scripts/assign_fixed_chains.py "
            f"--input_path {json_parsed} "
            f"--output_path {json_assigned} "
            f"--chain_list '{DESIGN_CHAIN}'")
    run_cmd(cmd2, cwd=MPNN_ROOT)

    # Step 3: Run neural network inference
    cmd3 = (f"python protein_mpnn_run.py "
            f"--jsonl_path {json_parsed} "
            f"--chain_id_jsonl {json_assigned} "
            f"--out_folder {OUTPUT_DIR} "
            f"--num_seq_per_target {NUM_SEQS} "
            f"--sampling_temp '{TEMPERATURE}' "
            f"--seed {SEED} "
            f"--batch_size {BATCH_SIZE}")
    run_cmd(cmd3, cwd=MPNN_ROOT)

if __name__ == "__main__":
    main()