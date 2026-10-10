import h5py
import torch 
from typing import Dict, Tuple

import sys 
sys.path.append('./src/')



@torch.jit.script
def treePredYhat(X: torch.Tensor, treeDepth: int, tree: Dict[str, torch.Tensor]) -> torch.Tensor:
    n, p = X.shape
    Tb = 2 ** treeDepth - 1
    t = torch.ones(n, device=X.device, dtype=torch.long)

    for iiii in range(treeDepth):
        decisions = (tree['a'][t - 1] * X).sum(dim=1) > tree['b'][t - 1]
        t = torch.where(decisions, 2 * t + 1, 2 * t).long()

    print("t-(Tb + 1): ", t - (Tb + 1))
    Yhat = tree['c'][(t - (Tb + 1)).long()]
    return Yhat 






@torch.jit.script
def treePredYhatLinear(X: torch.Tensor, treeDepth: int, tree: Dict[str, torch.Tensor]) -> torch.Tensor:
    n, p = X.shape
    Tb = 2 ** treeDepth - 1
    t = torch.ones(n, device=X.device, dtype=torch.long)

    for _ in range(treeDepth):
        decisions = (tree['a'][t - 1] * X).sum(dim=1) > tree['b'][t - 1]
        t = torch.where(decisions, 2 * t + 1, 2 * t).long()

    leaf_indices = (t - (Tb + 1)).long()
    k = tree['k'][leaf_indices]
    h = tree['h'][leaf_indices]
    Yhat = torch.bmm(k, X.unsqueeze(2)).squeeze(2) + h
    return Yhat


## Read Tree Path from HDF5 file
def readTreePath(treeDepth):
    indices_flags_dict = {}

    for treeDepthEach in range(treeDepth):
        treeDepthEach += 1
        with h5py.File("./src/DT/ancestorTF_File/ancestorTF_pairs_D"+str(treeDepthEach)+".hdf5", 'r') as ancestorTF_File:
            ancestorTF_pairs = ancestorTF_File['indicator_pairs'][:]
            ancestorTF_pairs_tensor = torch.tensor(ancestorTF_pairs, dtype=torch.long)
            ancestorTF_File.close()

        indices_tensor_long = ancestorTF_pairs_tensor[..., 0]
        flags_tensor_long = ancestorTF_pairs_tensor[..., 1]

        key = "D"+str(treeDepthEach)
        indices_flags_dict[key] = {
            'indices_tensor': indices_tensor_long,
            'flags_tensor': flags_tensor_long
        }

    return indices_flags_dict
