import logging
import math
import os
from datetime import datetime
from typing import Union, List
import dgl
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from dgl import function as fn

from commons.process_mols import AtomEncoder, rec_atom_feature_dims, rec_residue_feature_dims, lig_feature_dims
from commons.logger import log
from commons.utils import concat_if_list
from ImprovedEquibind import isomorphismTest,get_receptor_atom_subgraph

#from trainer.metrics import Rsquared, MeanPredictorLoss, MAE, PearsonR, RMSD, RMSDfraction, CentroidDist, \
   # CentroidDistFraction, RMSDmedian, CentroidDistMedian
#========================================================================================================================
class GraphNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, is_node=True):
        super().__init__()
        self.eps = eps
        self.num_features = num_features
        self.affine = affine
        self.is_node = is_node

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(self.num_features))
            self.beta = nn.Parameter(torch.zeros(self.num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def norm(self, x):
        mean = x.mean(dim=0, keepdim=True)
        var = x.std(dim=0, keepdim=True)
        return (x - mean) / (var + self.eps)

    def forward(self, g, h, node_type):
        graph_size = g.batch_num_nodes(node_type) if self.is_node else g.batch_num_edges(node_type)
        x_list = torch.split(h, graph_size.tolist())
        norm_list = []
        for x in x_list:
            norm_list.append(self.norm(x))
        norm_x = torch.cat(norm_list, 0)

        if self.affine:
            return self.gamma * norm_x + self.beta
        else:
            return norm_x

# =================================================================================================================       

class CoordsNorm(nn.Module):
    def __init__(self, eps=1e-8, scale_init=1.):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coords):
        norm = coords.norm(dim=-1, keepdim=True)
        normed_coords = coords / norm.clamp(min=self.eps)
        return normed_coords * self.scale

# =================================================================================================================            
def get_norm(layer_norm_type, dim):
    if layer_norm_type == 'BN':
        return nn.BatchNorm1d(dim)
    elif layer_norm_type == 'LN':
        return nn.LayerNorm(dim)
    elif layer_norm_type == 'GN':
        return GraphNorm(dim)
    else:
        assert layer_norm_type == '0' or layer_norm_type == 0
        return nn.Identity()
        
# =================================================================================================================       
def get_non_lin(type, negative_slope):
    if type == 'swish':
        return nn.SiLU()
    elif type == 'relu':
        return nn.ReLU()
    else:
        assert type == 'lkyrelu'
        return nn.LeakyReLU(negative_slope=negative_slope)

# =================================================================================================================       
def get_layer_norm(layer_norm_type, dim):
    if layer_norm_type == 'BN':
        return nn.BatchNorm1d(dim)
    elif layer_norm_type == 'LN':
        return nn.LayerNorm(dim)
    else:
        return nn.Identity()
# =================================================================================================================               
def get_dimofEmbedding(h_feats_rec ): # the H has dimension d x m
         d,m= h_feats_rec.size

         return(d,m)
# =================================================================================================================   
def apply_norm(g, h, norm_type, norm_layer):
    if norm_type == 'GN':
        return norm_layer(g, h)
    return norm_layer(h)
# =================================================================================================================   
def embeddings(use_rec_atoms,lig_graph,rec_graph,random_vec_dim,residue_emb_dim,use_scalar_features=True,num_lig_feats=None,separate_lig=False): 
    if use_rec_atoms:
        rec_embedder = AtomEncoder(emb_dim=residue_emb_dim - random_vec_dim,
                                        feature_dims=rec_atom_feature_dims, use_scalar_feat=use_scalar_features)
    else:
        rec_embedder = AtomEncoder(emb_dim=residue_emb_dim - random_vec_dim,
                                        feature_dims=rec_residue_feature_dims, use_scalar_feat=use_scalar_features)
        if use_rec_atoms:
            h_feats_rec = rec_embedder(rec_graph.ndata['feat'])
        else:
            h_feats_rec = rec_embedder(rec_graph.ndata['feat'])  # (N_res, emb_dim)
        return  h_feats_rec
# =================================================================================================================
def bindingsite(ligs_coords):
    centroid= torch.tensor(ligs_coords).mean()
    neighbour=[]
    for  lig_coords in zip( ligs_coords):
          temp = torch.cdist(lig_coords, centroid, p=2.0)
          if temp <= 20:
              neighbour.append(temp)
          temp=0
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
# def bindingSite(use_rec_atoms,lig_graph,rec_graph,num_att_heads,random_vec_dim,residue_emb_dim,use_scalar_features=True,num_lig_feats=None,separate_lig=False):
#      lig_atom_embedder = AtomEncoder(emb_dim=residue_emb_dim - random_vec_dim,
#                                              feature_dims=lig_feature_dims, use_scalar_feat=use_scalar_features,
#                                              n_feats_to_use=num_lig_feats)
#      h_feats_lig = lig_atom_embedder(lig_graph.ndata['feat'])  
#      ligs_node_idx = torch.cumsum(lig_graph.batch_num_nodes(), dim=0).tolist()
#      ligs_node_idx.insert(0, 0)
#      recs_node_idx = torch.cumsum(rec_graph.batch_num_nodes(), dim=0).tolist()
#      recs_node_idx.insert(0, 0)
#      h_feats_rec =  embeddings(use_rec_atoms,lig_graph,rec_graph,random_vec_dim,residue_emb_dim,use_scalar_features=True,num_lig_feats=None,separate_lig=False)
#      for idx in range(len(ligs_node_idx) - 1):
#         lig_start = ligs_node_idx[idx]
#         lig_end = ligs_node_idx[idx + 1]
#         rec_start = recs_node_idx[idx]
#         rec_end = recs_node_idx[idx + 1]
#         #ending of this part
#         rec_feats = h_feats_rec[rec_start:rec_end]  # (m, d)
#         rec_feats_mean = torch.mean(h_mean_rec(rec_feats), dim=0, keepdim=True)  # (1, d)
#         lig_feats = h_feats_lig[lig_start:lig_end]  # (n, d)
#         # _lig site = (keypts_attention_lig(lig_feats).view(-1, num_att_heads, d).transpose(0, 1) @
#         #                       keypts_queries_lig(rec_feats_mean).view(1, num_att_heads, d).transpose(0,
#         #                                                                       1).transpose(
#         #                           1, 2) /
#         #                       math.sqrt(d))
#      return

# class CentroidDist(nn.Module):
#     def __init__(self) -> None:
#         super(CentroidDist, self).__init__()

#     def forward(self, ligs_coords_pred: List[Tensor], ligs_coords: List[Tensor]) -> Tensor:
#         distances = []
#         for lig_coords_pred, lig_coords in zip(ligs_coords_pred, ligs_coords):
#                 distances.append(torch.linalg.norm(lig_coords_pred.mean(dim=0)-lig_coords.mean(dim=0)))
#         return torch.tensor(distances).mean()

#==========================================================================================================================================================================================================================
class MBP_Layer(nn.Module):
    def __init__(self,out_feats_dim,rec_graph,rec_input_edge_feats_dim,
        debug,
        device,
        dropout,
        layer_norm,
        standard_norm_order,
        nonlin, 
        final_h_layer_norm,
        leakyrelu_neg_slope,
        layer_norm_coords,
        x_connection_init,
        
    
    ):

        super(MBP_Layer, self).__init__()
        
         # the hidden feature diminsion
        self.rec_input_edge_feats_dim=rec_input_edge_feats_dim
        self.rec_graph=rec_graph
        self.debug = debug
        self.device=device
        self.device=device
        self.dropout=dropout
        self.out_feats_dim=out_feats_dim
        self.standard_norm_order=standard_norm_order
        self.x_connection_init=x_connection_init
        self.final_h_layer_norm=final_h_layer_norm
        outputs = self.iegmn( rec_graph,0)
        h_feats_rec= outputs
        m, in_dim= h_feats_rec.shape
        rec_edge_mlp_input_dim = (in_dim* 2) 
        

        #Edges layer
        if self.standard_norm_order:
            self.rec_edge_mlp = nn.Sequential(
                nn.Linear(rec_edge_mlp_input_dim, in_dim),
                get_layer_norm(layer_norm, in_dim),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                nn.Linear(in_dim, in_dim),
                get_layer_norm(layer_norm, in_dim),
            )
        else:
            self.rec_edge_mlp = nn.Sequential(
                nn.Linear(rec_edge_mlp_input_dim, in_dim),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                get_layer_norm(layer_norm, in_dim),
                nn.Linear(in_dim, in_dim),
            )

       #defining nodes of GNN layer
        self.node_norm = nn.Identity()
        if self.normalize_coordinate_update:
            self.rec_feats_norm = h_feats_rec(scale_init=1e-2)
        
        # if self.normalize_coordinate_update:
        #     self.rec_coords_norm =h_feats_rec(scale_init=1e-2)
        self.att_mlp_Q = nn.Sequential(
            nn.Linear(in_dim, in_dim, bias=False),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.att_mlp_K = nn.Sequential(
            nn.Linear(in_dim, in_dim, bias=False),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.att_mlp_V = nn.Sequential(
            nn.Linear(in_dim, in_dim, bias=False),
        )
       #Nodes layer
        if self.standard_norm_order:
            self.node_mlp = nn.Sequential(
                nn.Linear(rec_edge_mlp_input_dim, in_dim),
                get_layer_norm(layer_norm, in_dim),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                nn.Linear(in_dim, in_dim),
                get_layer_norm(layer_norm, in_dim),
            )
        else:
            self.node_mlp = nn.Sequential(
                nn.Linear(rec_edge_mlp_input_dim, in_dim),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                get_layer_norm(layer_norm, in_dim),
                nn.Linear(in_dim, in_dim),
            )
       
        self.final_h_layernorm_layer = get_norm(self.final_h_layer_norm,in_dim)
    

        if self.standard_norm_order:
            self.feat_mlp_rec = nn.Sequential(
                nn.Linear(rec_edge_mlp_input_dim, in_dim),
                get_layer_norm(layer_norm_coords, in_dim),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                nn.Linear(in_dim, 1)
            )
        else:
            self.feat_mlp_rec = nn.Sequential(
                nn.Linear(rec_edge_mlp_input_dim, in_dim),
                get_non_lin(nonlin, leakyrelu_neg_slope),
                get_layer_norm(layer_norm_coords, in_dim),
                nn.Linear(in_dim, 1)
            )
       
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p, gain=1.)
            else:
                torch.nn.init.zeros_(p)
    def apply_edges_rec(self, edges):
        if self.use_dist_in_layers and self.rec_evolve:
            squared_distance = torch.sum(edges.data['x_rel'] ** 2, dim=1, keepdim=True)
            # divide square distance by 10 to have a nicer separation instead of many 0.00000
            x_rel_mag = torch.cat([torch.exp(-(squared_distance / self.rec_square_distance_scale) / sigma) for sigma in
                                   self.all_sigmas_dist], dim=-1)
            return {'msg': self.rec_edge_mlp(
                torch.cat([edges.src['feat'], edges.dst['feat'], edges.data['feat'], x_rel_mag], dim=1))}
        else:
            return {
                'msg': self.rec_edge_mlp(torch.cat([edges.src['feat'], edges.dst['feat'], edges.data['feat']], dim=1))}
    def forward(self, rec_graph,coords_rec, h_feats_rec, original_receptor_node_features, geometry_graph):
        with  rec_graph.local_scope():
            rec_graph.ndata['x_now'] = coords_rec
            rec_graph.ndata['feat'] = h_feats_rec 
            rec_graph.apply_edges(self.apply_edges_rec)
            #h_feats_rec_norm = apply_norm(rec_graph, h_feats_rec, self.final_h_layer_norm, self.final_h_layernorm_layer) 
            #x_evolved_rec = coords_rec
            rec_graph.update_all(fn.copy_edge('msg', 'm'), fn.mean('m', 'aggr_msg'))
            input_node_upd_receptor = torch.cat((self.node_norm(rec_graph.ndata['feat']),
                                                 rec_graph.ndata['aggr_msg'],
                                                 original_receptor_node_features), dim=-1)

            # Skip connections
            if self.h_feats_dim == self.out_feats_dim:
    
                node_upd_receptor = self.skip_weight_h * self.node_mlp(input_node_upd_receptor) + (
                        1. - self.skip_weight_h) * h_feats_rec
            else:
                
                node_upd_receptor = self.node_mlp(input_node_upd_receptor)

            if self.debug:
                log('node_mlp params')
                for p in self.node_mlp.parameters():
                    log(torch.max(p.abs()), 'max node_mlp_params')
                    log(torch.min(p.abs()), 'min of abs node_mlp_params')

            node_upd_receptor = apply_norm(rec_graph, node_upd_receptor,
                                           self.final_h_layer_norm, self.final_h_layernorm_layer)
            return  node_upd_receptor,rec_graph
    def __repr__(self):
        return "MLP Layer " + str(self.__dict__)

class MBP(nn.Module):

    def __init__(self, n_lays, debug, device, use_rec_atoms,nonlin,ligs_coords, leakyrelu_neg_slope,shared_layers,rec_graph,random_vec_dim,residue_emb_dim,use_scalar_features=True, **kwargs):
        super(MBP, self).__init__()
        self.debug = debug
        self.device = device
        self.use_rec_atoms = use_rec_atoms
        self.ligs_coords= ligs_coords
        self.rec_graph= rec_graph
        if self.use_rec_atoms:
            rec_embedder = AtomEncoder(emb_dim=residue_emb_dim - self.random_vec_dim,
                                            feature_dims=rec_atom_feature_dims, use_scalar_feat=use_scalar_features)
        else:
            rec_embedder = AtomEncoder(emb_dim=residue_emb_dim - self.random_vec_dim,
                                            feature_dims=rec_residue_feature_dims, use_scalar_feat=use_scalar_features)
            if use_rec_atoms:
                 h_feats_rec = rec_embedder(rec_graph.ndata['feat'])
            else:
                 h_feats_rec = rec_embedder(rec_graph.ndata['feat'])  # (N_res, emb_dim)
        self.mlp_Layer = nn.ModuleList()
        m, input_node_feats_dim =h_feats_rec.shape
        self.mlp_Layer.append(
            MBP_Layer(orig_h_feats_dim=input_node_feats_dim,
                        h_feats_dim=input_node_feats_dim,
                        out_feats_dim=input_node_feats_dim,
                        nonlin=nonlin,
                        leakyrelu_neg_slope=leakyrelu_neg_slope,
                        debug=debug,
                        device=device,
                        **kwargs))

        if shared_layers:
            interm_lay = MBP_Layer(orig_h_feats_dim=input_node_feats_dim,
                                     h_feats_dim=input_node_feats_dim,
                                     out_feats_dim=input_node_feats_dim,
                                     nonlin=nonlin,
                                     leakyrelu_neg_slope=leakyrelu_neg_slope,
                                     debug=debug,
                                     device=device,
                                     **kwargs)
            for layer_idx in range(1, n_lays):
                self.mlp_Layer.append(interm_lay)
        else:
            for layer_idx in range(1, n_lays):
                debug_this_layer = debug if n_lays - 1 == layer_idx else False
                self.mlp_Layer.append(
                    MBP_Layer(orig_h_feats_dim=input_node_feats_dim,
                                h_feats_dim=input_node_feats_dim,
                                out_feats_dim=input_node_feats_dim,
                                nonlin=nonlin,
                                leakyrelu_neg_slope=leakyrelu_neg_slope,
                                debug=debug_this_layer,
                                device=device,
                                **kwargs))
       
        
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p, gain=1.)
            else:
                torch.nn.init.zeros_(p)
                
    def forward(self,rec_graph,h_feats_rec, ligs_coords,rec_coords,lig,rec,lig_graph):
        h_feats_rec_norm =  torch.nn.functional.normalize(h_feats_rec, p=2.0,  eps=1e-12, out=None)
        coords_rec = rec_graph.ndata['x']
        #coords_rec =torch.nn.functional.normalize(coords_rec, p=2.0,  eps=1e-12, out=None)
        original_receptor_node_features = h_feats_rec   # should I use it or should I normalize it first
        orig_coords_rec = rec_graph.ndata['x']
        #orig_coords_rec  =torch.nn.functional.normalize(orig_coords_rec  p=2.0,  eps=1e-12, out=None)
        for i, layer in enumerate(self.mlp_Layer):
            if self.debug: log('layer ', i)
            h_feats_rec = layer(
                                rec_graph=rec_graph,
                                coords_rec=coords_rec,
                                h_feats_rec=h_feats_rec_norm,
                                original_receptor_node_features=original_receptor_node_features,
                                orig_coords_rec=orig_coords_rec,
                                
                                )
        SimilarityMatrix=symmetric_matrix(h_feats_rec_norm)
        listOfNeighbour= bindingsite(ligs_coords)
        for neighbour in range(listOfNeighbour):
            AdditiveBindingSite=[]
            for idx in range(SimilarityMatrix.shape):
                #if neighbour==
                if SimilarityMatrix[idx]>=0.98:  # tolenance range, instead of 1.0
                    AdditiveBindingSite= AdditiveBindingSite.append(idx)   # I am getting index in similarity matrix, I need to map it to rec. graph 
        #how to link the two techniques, graph & simmilarity matrix
        rec_subgraph = get_receptor_atom_subgraph(rec, rec_coords, lig,  lig_coords=None ,graph_cutoff=4, max_neighbor=10, subgraph_radius=20)  # how to check if they are the same graph 
        isomorphismFlag = isomorphismTest(rec_subgraph,lig_graph)
        return AdditiveBindingSite, isomorphismFlag
    
class EquiBind2(nn.Module):

    def __init__(self, device='cuda:0', debug=False, use_evolved_lig=False, evolve_only=False, **kwargs):
        super(EquiBind2, self).__init__()
        self.debug = debug
        self.evolve_only = evolve_only
        self.device = device
        self.MBP = MBP(debug=self.debug,device=self.device, **kwargs)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p, gain=1.)
            else:
                torch.nn.init.zeros_(p)

    def forward(self, rec_graph,h_feats_rec, ligs_coords,rec_coords,lig,rec,lig_graph,complex_names=None):
        if self.debug: log(complex_names)
        outputs = self.MBP(rec_graph,h_feats_rec, ligs_coords,rec_coords,lig,rec,lig_graph)
        return outputs[0], outputs[1]

    def __repr__(self):
        return "EquiBind2 " + str(self.__dict__)