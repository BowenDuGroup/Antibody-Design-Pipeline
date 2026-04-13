import os
import glob
import torch
import numpy as np
import re
from torch.utils.data import Dataset
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
import warnings
from Bio import BiopythonWarning

warnings.simplefilter('ignore', BiopythonWarning)

RESTYPE_ORDER = {k: i for i, k in enumerate('ARNDCQEGHILKMFPSTWYV')}

# Fallback motifs for CDR identification
CDR_MOTIFS = [
    r'C[A-Z]{1,15}[WY][A-Z]{1,15}W',
    r'YYC[A-Z]{1,30}WG.G',
    r'YYC[A-Z]{1,30}FG.G',
]

class AntibodyComplexDataset(Dataset):
    def __init__(self, pdb_dir):
        self.pdb_files = glob.glob(os.path.join(pdb_dir, "*.pdb"))
        self.parser = PDBParser(QUIET=True)

    def __len__(self):
        return len(self.pdb_files)

    def __getitem__(self, idx):
        pdb_path = self.pdb_files[idx]
        structure = self.parser.get_structure('complex', pdb_path)
        
        all_xyz = []
        all_seq = []
        all_mask = []

        for model in structure:
            for chain in model:
                residues = [r for r in chain if r.id[0] == ' ']
                seq_str = ''.join([seq1(r.get_resname(), undef_code='X') for r in residues])
                is_cdr = [False] * len(seq_str)
                
                # CDR matching
                for motif in CDR_MOTIFS:
                    for match in re.finditer(motif, seq_str):
                        s, e = match.span()
                        for i in range(s + 3, e - 3):
                            if i < len(is_cdr): 
                                is_cdr[i] = True
                
                # Coordinate extraction (N, CA, C)
                for i, res in enumerate(residues):
                    all_seq.append(RESTYPE_ORDER.get(seq1(res.get_resname(), undef_code='X'), 20))
                    all_mask.append(is_cdr[i])
                    
                    atom_coords = []
                    for atom_name in ['N', 'CA', 'C']:
                        if atom_name in res:
                            atom_coords.append(res[atom_name].get_coord())
                        else:
                            # Fallback to CA coordinates if missing
                            atom_coords.append(res['CA'].get_coord() if 'CA' in res else np.zeros(3))
                    
                    all_xyz.append(atom_coords)

            break 

        return {
            'xyz': torch.tensor(np.array(all_xyz), dtype=torch.float32),
            'seq': torch.tensor(all_seq, dtype=torch.long),
            'mask': torch.tensor(all_mask, dtype=torch.bool)
        }