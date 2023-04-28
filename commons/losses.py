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


# Ligand residue locations: a_i in R^3. Receptor: b_j in R^3
# Ligand: G_l(x) = -sigma * ln( \sum_i  exp(- ||x - a_i||^2 / sigma)  ), same for G_r(x)
# Ligand surface: x such that G_l(x) = surface_ct
# Other properties: G_l(a_i) < 0, G_l(x) = infinity if x is far from all a_i
# Intersection of ligand and receptor: points x such that G_l(x) < surface_ct && G_r(x) < surface_ct
# Intersection loss: IL = \avg_i max(0, surface_ct - G_r(a_i)) + \avg_j max(0, surface_ct - G_l(b_j))
def G_fn(protein_coords, x, sigma):
    # protein_coords: (n,3) ,  x: (m,3), output: (m,)
    e = torch.exp(- torch.sum((protein_coords.view(1, -1, 3) - x.view(-1, 1, 3)) ** 2, dim=2) / float(sigma))  # (m, n)
    return - sigma * torch.log(1e-3 + e.sum(dim=1))


def compute_body_intersection_loss(model_ligand_coors_deform, bound_receptor_repres_nodes_loc_array, sigma, surface_ct):
    loss = torch.mean(
        torch.clamp(surface_ct - G_fn(bound_receptor_repres_nodes_loc_array, model_ligand_coors_deform, sigma),
                    min=0)) + \
           torch.mean(
               torch.clamp(surface_ct - G_fn(model_ligand_coors_deform, bound_receptor_repres_nodes_loc_array, sigma),
                           min=0))
    return loss


def compute_sq_dist_mat(X_1, X_2):
    '''Computes the l2 squared cost matrix between two point cloud inputs.
    Args:
        X_1: [n, #features] point cloud, tensor
        X_2: [m, #features] point cloud, tensor
    Output:
        [n, m] matrix of the l2 distance between point pairs
    '''
    n_1, _ = X_1.size()
    n_2, _ = X_2.size()
    X_1 = X_1.view(n_1, 1, -1)
    X_2 = X_2.view(1, n_2, -1)
    squared_dist = (X_1 - X_2) ** 2
    cost_mat = torch.sum(squared_dist, dim=2)
    return cost_mat


def compute_ot_emd(cost_mat, device):
    cost_mat_detach = cost_mat.detach().cpu().numpy()
    a = np.ones([cost_mat.shape[0]]) / cost_mat.shape[0]
    b = np.ones([cost_mat.shape[1]]) / cost_mat.shape[1]
    ot_mat = ot.emd(a=a, b=b, M=cost_mat_detach, numItermax=10000)
    ot_mat_attached = torch.tensor(ot_mat, device=device, requires_grad=False).float()
    ot_dist = torch.sum(ot_mat_attached * cost_mat)
    return ot_dist, ot_mat_attached


def compute_revised_intersection_loss(lig_coords, rec_coords, alpha = 0.2, beta=8, aggression=0):
    distances = compute_sq_dist_mat(lig_coords,rec_coords)
    if aggression > 0:
        aggression_term = torch.clamp(-torch.log(torch.sqrt(distances)/aggression+0.01), min=1)
    else:
        aggression_term = 1
    distance_losses = aggression_term * torch.exp(-alpha*distances * torch.clamp(distances*4-beta, min=1))
    return distance_losses.sum()

class BindingLoss(_Loss):
    def __init__(self, ot_loss_weight=1, intersection_loss_weight=0, intersection_sigma=0, geom_reg_loss_weight=1, loss_rescale=True,
                 intersection_surface_ct=0, key_point_alignmen_loss_weight=0,revised_intersection_loss_weight=0, centroid_loss_weight=0, kabsch_rmsd_weight=0,translated_lig_kpt_ot_loss=False, revised_intersection_alpha=0.1, revised_intersection_beta=8, aggression=0) -> None:
        super(BindingLoss, self).__init__()
        self.ot_loss_weight = ot_loss_weight
        self.intersection_loss_weight = intersection_loss_weight
        self.intersection_sigma = intersection_sigma
        self.revised_intersection_loss_weight =revised_intersection_loss_weight
        self.intersection_surface_ct = intersection_surface_ct
        self.key_point_alignmen_loss_weight = key_point_alignmen_loss_weight
        self.centroid_loss_weight = centroid_loss_weight
        self.translated_lig_kpt_ot_loss= translated_lig_kpt_ot_loss
        self.kabsch_rmsd_weight = kabsch_rmsd_weight
        self.revised_intersection_alpha = revised_intersection_alpha
        self.revised_intersection_beta = revised_intersection_beta
        self.aggression =aggression
        self.loss_rescale = loss_rescale
        self.geom_reg_loss_weight = geom_reg_loss_weight
        self.mse_loss = MSELoss()

    def forward(self, h_feats_rec, binding_sites, gold_centers, **kwargs):
        # Compute MSE loss for each protein individually, then average over the minibatch.
        loss = 0
        for i in range(len(h_feats_rec)):
            normalized_feat = F.normalize(h_feats_rec[i])
            cosine = torch.matmul(normalized_feat, normalized_feat.t())
            this_loss = (cosine + 1).mean()
            loss += this_loss
        loss = loss / len(h_feats_rec)
        loss_components = {"loss": loss}
        return loss, loss_components

class TorsionLoss(_Loss):
    def __init__(self) -> None:
        super(TorsionLoss, self).__init__()
        self.mse_loss = MSELoss()

    def forward(self, angles_pred, angles, masks, **kwargs):
        return self.mse_loss(angles_pred*masks,angles*masks)