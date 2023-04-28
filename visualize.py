#%%
import argparse
import sys

from copy import deepcopy
import os
from dgl import load_graphs, save_graphs
from rdkit import Chem
from rdkit.Chem import RemoveHs
from rdkit.Geometry import Point3D
from tqdm import tqdm
from datasets.pdbbind import PDBBind
from Bio.PDB import *
from joblib import Parallel, delayed, cpu_count
import torch
import dgl
from biopandas.pdb import PandasPdb
from joblib.externals.loky import get_reusable_executor
import nglview as nv
import py3Dmol
filepath=os.path.join("gs","l10gs_protein_processed.pdb")
view1= nv.show_file(filepath)
view1
view1.clear_representations()
view1.add_representation("hyperball","ligand", color="blue", opacity=0.9)
view1.add_representation("cartoon","protein", color="red", opacity=0.9)
view1.center("ligand")


# %%
