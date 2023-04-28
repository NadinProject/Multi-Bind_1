
import torch
from datasets.samplers import HardSampler
from trainer.trainer import Trainer


class BindingTrainer(Trainer):
    def __init__(self, **kwargs):
        super(BindingTrainer, self).__init__(**kwargs)

    def forward_pass(self, batch):
        lig_graphs, rec_graphs, ligs_coords, recs_coords, ligs_pocket_coords, recs_pocket_coords, geometry_graphs, complex_names = tuple(
            batch)
        h_feats_rec, binding_sites,batches, gold_centers = self.model(lig_graphs, rec_graphs, geometry_graphs,
                                                                                         complex_names=complex_names,
                                                                                         epoch=self.epoch)
        loss, loss_components = self.loss_func(h_feats_rec, binding_sites, gold_centers)
        #return loss, loss_components, None, ligs_coords
        return loss, loss_components, None, ligs_coords
    def MultipleBindingSite(self, batch):
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
                  
        return  Multiple_binding_site
    def after_batch(self, ligs_coords_pred, ligs_coords, batch_indices):
        cutoff = 5
        centroid_distances = []
        for lig_coords_pred, lig_coords in zip(ligs_coords_pred, ligs_coords):
            centroid_distances.append(torch.linalg.norm(lig_coords_pred.mean(dim=0) - lig_coords.mean(dim=0)))
        centroid_distances = torch.tensor(centroid_distances)
        above_cutoff = torch.tensor(batch_indices)[torch.where(centroid_distances > cutoff)[0]]
        if isinstance(self.sampler, HardSampler):
            self.sampler.add_hard_indices(above_cutoff.tolist())

    def after_epoch(self):
        if isinstance(self.sampler, HardSampler):
            self.sampler.set_hard_indices()
