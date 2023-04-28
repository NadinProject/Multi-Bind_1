import itertools
import math

import dgl
import ot
import torch
from torch import Tensor, nn
from torch.distributions import MultivariateNormal
from torch.nn.modules.loss import _Loss, L1Loss, MSELoss, BCEWithLogitsLoss
import numpy as np
import torch.nn.functional as F
class MultipleBindingSite():
    def __init__(self, **kwargs):
        super(MultipleBindingSite, self).__init__(**kwargs)
        

    def forward(self, batch):
        lig_graphs, rec_graphs, ligs_coords, recs_coords, ligs_pocket_coords, recs_pocket_coords, geometry_graphs, complex_names = tuple(
            batch)
        h_feats_rec, binding_sites,batches, gold_centers = self.model(lig_graphs, rec_graphs, geometry_graphs,
                                                                                         complex_names=complex_names,
                                                                                         epoch=self.epoch)
        h_feats_rec_norm= list()
        h_feats_rec_t_norm= list()
        S_matrix=list()
        h_feats_batch=  torch.split(h_feats_rec, batches)
        for i in range(len(batches)):
             h_feats_rec_norm.append(torch.nn.functional.normalize(h_feats_batch[i], p=2.0,  eps=1e-12, out=None))
             h_feats_rec_t_norm.append(torch.transpose(h_feats_rec_norm[i], 0, 1))
             S_matrix.append((torch.mm(h_feats_rec_norm[i], h_feats_rec_t_norm[i])+1)/2)
          
        Multiple_binding_site=list()
        for batch in range(batches):
            for i in range(len(binding_sites[batch])):
                Indication_list=list()
                Cuurent_similarity_matrix=S_matrix[i]
                for j in range(len(S_matrix[i])):
                    if(Cuurent_similarity_matrix[j])==1:
                        Indication_list.append(j)
                if (len(Indication_list)!=1):
                    break;
                else:
                    Multiple_binding_site[i]= Indication_list
                  
        return MultipleBindingSite