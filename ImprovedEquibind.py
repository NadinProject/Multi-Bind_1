import logging
import math
import os
from datetime import datetime

import dgl
import torch
from torch import nn
from dgl import function as fn

from commons.process_mols import AtomEncoder, rec_atom_feature_dims, rec_residue_feature_dims, lig_feature_dims
from commons.logger import log
import warnings

import pandas as pd
import numpy as np
import scipy.spatial as spa
from Bio.PDB import get_surface, PDBParser, ShrakeRupley
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from biopandas.pdb import PandasPdb
from rdkit import Chem
from rdkit.Chem import MolFromPDBFile, AllChem, GetPeriodicTable, rdDistGeom
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from scipy import spatial
from scipy.special import softmax

#from geometry_utils import rigid_transform_Kabsch_3D, rigid_transform_Kabsch_3D_torch
from commons.logger import log
def bindingsite(ligs_coords):
    centroid= torch.tensor(ligs_coords).mean()
    neighbour=[]
    for  lig_coords in zip( ligs_coords):
          temp = torch.cdist(lig_coords, centroid, p=2.0)
          if temp <= 20:
              neighbour= temp
    return  neighbour
# =================================================================================================================
def cosine_distance(h_feats_rec):
    h_feats_rec_t = torch.transpose(h_feats_rec)
    h_feats_rec_norm =  torch.nn.functional.normalize(h_feats_rec, p=2.0,  eps=1e-12, out=None)
    h_feats_rec_norm_t =  torch.nn.functional.normalize(h_feats_rec_t, p=2.0, eps=1e-12, out=None)
   
    c_distance= torch.mm(h_feats_rec, h_feats_rec_t)/(h_feats_rec_norm * h_feats_rec_norm_t)
    return c_distance
# =================================================================================================================
def loss_function(c_distance):
    c_distance[c_distance<0]=0
    return 1-c_distance
    #return 2-c_distance
 # =================================================================================================================   
def symmetric_matrix(h_feats_rec_norm):
    h_feats_rec_norm_t = torch.transpose(h_feats_rec_norm)
    S_matrix= torch.mm(h_feats_rec_norm, h_feats_rec_norm_t)+1
    return S_matrix
# =================================================================================================================
def loss_function_f(S_matrix):
    #  temp = torch.trace(S_matrix)
    #  f_norm = torch.sqrt(temp)
     f_norm = torch.norm(S_matrix,'fro') 
     return f_norm 
# =================================================================================================================
def isomorphismTest(g,h):
     flag=True
     e1=g.edges
     e2=h.edges
     if (g.number_of_edges) != (h.number_of_edges):
        return False
     else:
         for edge in e1:
            if edge not in e2:
                return False
     return flag
# =================================================================================================================
sr = ShrakeRupley(probe_radius=1.4,  # in A. Default is 1.40 roughly the radius of a water molecule.
                  n_points=100)  # resolution of the surface of each atom. Default is 100. A higher number of points results in more precise measurements, but slows down the calculation.
def get_receptor_atom_subgraph(rec, rec_coords, lig,  lig_coords=None ,graph_cutoff=4, max_neighbor=10, subgraph_radius=20):
    lig_coords = lig.GetConformer().GetPositions() if lig_coords == None else lig_coords
    rec_coords = np.concatenate(rec_coords, axis=0)
    sr.compute(rec, level="A")
    lig_rec_distance = spa.distance.cdist(lig_coords, rec_coords)
    subgraph_indices = np.where(np.min(lig_rec_distance, axis=0) < subgraph_radius)[0]
    subgraph_coords = rec_coords[subgraph_indices]
    distances = spa.distance.cdist(subgraph_coords, subgraph_coords)
    src_list = []
    dst_list = []
    dist_list = []
    mean_norm_list = []
    for i in range(len(subgraph_coords)):
        dst = list(np.where(distances[i, :] < graph_cutoff)[0])
        dst.remove(i)
        if max_neighbor != None and len(dst) > max_neighbor:
            dst = list(np.argsort(distances[i, :]))[1: max_neighbor + 1]
        if len(dst) == 0:
            dst = list(np.argsort(distances[i, :]))[1:2]  # choose second because first is i itself
            log(f'The graph_cutoff {graph_cutoff} was too small for one c_alpha such that it had no neighbors. So we connected it to the closest other c_alpha')
        assert i not in dst
        src = [i] * len(dst)
        src_list.extend(src)
        dst_list.extend(dst)
        valid_dist = list(distances[i, dst])
        dist_list.extend(valid_dist)
        valid_dist_np = distances[i, dst]
        sigma = np.array([1., 2., 5., 10., 30.]).reshape((-1, 1))
        weights = softmax(- valid_dist_np.reshape((1, -1)) ** 2 / sigma, axis=1)  # (sigma_num, neigh_num)
        assert weights[0].sum() > 1 - 1e-2 and weights[0].sum() < 1.01
        diff_vecs = subgraph_coords[src, :] - subgraph_coords[dst, :]  # (neigh_num, 3)
        mean_vec = weights.dot(diff_vecs)  # (sigma_num, 3)
        denominator = weights.dot(np.linalg.norm(diff_vecs, axis=1))  # (sigma_num,)
        mean_vec_ratio_norm = np.linalg.norm(mean_vec, axis=1) / denominator  # (sigma_num,)
        mean_norm_list.append(mean_vec_ratio_norm)
    assert len(src_list) == len(dst_list)
    assert len(dist_list) == len(dst_list)
    graph = dgl.graph((torch.tensor(src_list), torch.tensor(dst_list)), num_nodes=len(subgraph_coords), idtype=torch.int32)

    graph.ndata['x'] = torch.from_numpy(subgraph_coords.astype(np.float32))
    log('number of subgraph nodes = ', len(subgraph_coords), ' number of edges in subgraph = ', len(dist_list) )
    return graph
def Subgraph (lig_graph, rec_graph, max_neighbor=10, subgraph_radius=20 ):
     return

