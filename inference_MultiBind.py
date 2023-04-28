import argparse
import sys

from copy import deepcopy

import os

from dgl import load_graphs

from rdkit import Chem
from rdkit.Chem import RemoveHs
from rdkit.Geometry import Point3D
from tqdm import tqdm

from commons.geometry_utils import rigid_transform_Kabsch_3D, get_torsions, get_dihedral_vonMises, apply_changes
from commons.logger import Logger
from commons.process_mols import read_molecule, get_lig_graph_revised, \
    get_rec_graph, get_geometry_graph, get_geometry_graph_ring, \
    get_receptor_inference

from train import load_model

from datasets.pdbbind import PDBBind

from commons.utils import seed_all, read_strings_from_txt

import yaml

from datasets.custom_collate import *  # do not remove
from models import *  # do not remove
from torch.nn import *  # do not remove
from torch.optim import *  # do not remove
from commons.losses import *  # do not remove
from torch.optim.lr_scheduler import *  # do not remove

from torch.utils.data import DataLoader

from trainer.metrics import Rsquared, MeanPredictorLoss, MAE, PearsonR, RMSD, RMSDfraction, CentroidDist, \
    CentroidDistFraction, RMSDmedian, CentroidDistMedian

# turn on for debugging C code like Segmentation Faults
import faulthandler

faulthandler.enable()


def parse_arguments(arglist = None):
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs_clean/inference.yml')
    p.add_argument('--checkpoint', type=str, help='path to .pt file in a checkpoint directory')
    p.add_argument('--output_directory', type=str, default=None, help='path where to put the predicted results')
    p.add_argument('--run_corrections', type=bool, default=False,
                   help='whether or not to run the fast point cloud ligand fitting')
    p.add_argument('--run_dirs', type=list, default=[], help='path directory with saved runs')
    p.add_argument('--fine_tune_dirs', type=list, default=[], help='path directory with saved finetuning runs')
    p.add_argument('--inference_path', type=str, help='path to some pdb files for which you want to run inference')
    p.add_argument('--experiment_name', type=str, help='name that will be added to the runs folder output')
    p.add_argument('--logdir', type=str, default='runs', help='tensorboard logdirectory')
    p.add_argument('--num_epochs', type=int, default=2500, help='number of times to iterate through all samples')
    p.add_argument('--batch_size', type=int, default=1024, help='samples that will be processed in parallel')
    p.add_argument('--patience', type=int, default=20, help='stop training after no improvement in this many epochs')
    p.add_argument('--minimum_epochs', type=int, default=0, help='minimum numer of epochs to run')
    p.add_argument('--dataset_params', type=dict, default={},
                   help='parameters with keywords of the dataset')
    p.add_argument('--num_train', type=int, default=-1, help='n samples of the model samples to use for train')
    p.add_argument('--num_val', type=int, default=None, help='n samples of the model samples to use for validation')
    p.add_argument('--seed', type=int, default=1, help='seed for reproducibility')
    p.add_argument('--multithreaded_seeds', type=list, default=[],
                   help='if this is non empty, multiple threads will be started, training the same model but with the different seeds')
    p.add_argument('--seed_data', type=int, default=1, help='if you want to use a different seed for the datasplit')
    p.add_argument('--loss_func', type=str, default='MSELoss', help='Class name of torch.nn like [MSELoss, L1Loss]')
    p.add_argument('--loss_params', type=dict, default={}, help='parameters with keywords of the chosen loss function')
    p.add_argument('--optimizer', type=str, default='Adam', help='Class name of torch.optim like [Adam, SGD, AdamW]')
    p.add_argument('--optimizer_params', type=dict, help='parameters with keywords of the chosen optimizer like lr')
    p.add_argument('--clip_grad', type=float, default=None, help='clip gradients if magnitude is greater')
    p.add_argument('--lr_scheduler', type=str,
                   help='Class name of torch.optim.lr_scheduler like [CosineAnnealingLR, ExponentialLR, LambdaLR]')
    p.add_argument('--lr_scheduler_params', type=dict, help='parameters with keywords of the chosen lr_scheduler')
    p.add_argument('--scheduler_step_per_batch', default=True, type=bool,
                   help='step every batch if true step every epoch otherwise')
    p.add_argument('--log_iterations', type=int, default=-1,
                   help='log every log_iterations iterations (-1 for only logging after each epoch)')
    p.add_argument('--expensive_log_iterations', type=int, default=100,
                   help='frequency with which to do expensive logging operations')
    p.add_argument('--eval_per_epochs', type=int, default=0,
                   help='frequency with which to do run the function run_eval_per_epoch that can do some expensive calculations on the val set or sth like that. If this is zero, then the function will never be called')
    p.add_argument('--metrics', default=[], help='tensorboard metrics [mae, mae_denormalized, qm9_properties ...]')
    p.add_argument('--main_metric', default='loss', help='for early stopping etc.')
    p.add_argument('--main_metric_goal', type=str, default='min', help='controls early stopping. [max, min]')
    p.add_argument('--val_per_batch', type=bool, default=True,
                   help='run evaluation every batch and then average over the eval results. When running the molhiv benchmark for example, this needs to be Fale because we need to evaluate on all val data at once since the metric is rocauc')
    p.add_argument('--tensorboard_functions', default=[], help='choices of the TENSORBOARD_FUNCTIONS in utils')
    p.add_argument('--num_epochs_local_only', type=int, default=1,
                   help='when training with OptimalTransportTrainer, this specifies for how many epochs only the local predictions will get a loss')

    p.add_argument('--collate_function', default='graph_collate', help='the collate function to use for DataLoader')
    p.add_argument('--collate_params', type=dict, default={},
                   help='parameters with keywords of the chosen collate function')
    p.add_argument('--device', type=str, default='cuda', help='What device to train on: cuda or cpu')

    p.add_argument('--models_to_save', type=list, default=[],
                   help='specify after which epochs to remember the best model')

    p.add_argument('--model_type', type=str, default='MPNN', help='Classname of one of the models in the models dir')
    p.add_argument('--model_parameters', type=dict, help='dictionary of model parameters')

    p.add_argument('--trainer', type=str, default='binding', help='')
    p.add_argument('--train_sampler', type=str, default=None, help='any of pytorchs samplers or a custom sampler')

    p.add_argument('--eval_on_test', type=bool, default=True, help='runs evaluation on test set if true')
    p.add_argument('--check_se3_invariance', type=bool, default=False, help='check it instead of generating files')
    p.add_argument('--num_confs', type=int, default=1, help='num_confs if using rdkit conformers')
    p.add_argument('--use_rdkit_coords', action="store_true",
                   help='override the rkdit usage behavior of the used model')
    p.add_argument('--no_use_rdkit_coords', action="store_false", dest = "use_rdkit_coords",
                   help='override the rkdit usage behavior of the used model')

    cmdline_parser = deepcopy(p)
    args = p.parse_args(arglist)
    clear_defaults = {key: argparse.SUPPRESS for key in args.__dict__}
    cmdline_parser.set_defaults(**clear_defaults)
    cmdline_parser._defaults = {}
    cmdline_args = cmdline_parser.parse_args(arglist)
    
    return args, cmdline_args


def inference(args, tune_args=None):
    sys.stdout = Logger(logpath=os.path.join(os.path.dirname(args.checkpoint), f'inference.log'), syspart=sys.stdout)
    sys.stderr = Logger(logpath=os.path.join(os.path.dirname(args.checkpoint), f'inference.log'), syspart=sys.stderr)
    seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")

    use_rdkit_coords = args.dataset_params[
        'use_rdkit_coords'] if 'use_rdkit_coords' in args.dataset_params.keys() else False
    data = PDBBind(device=device, complex_names_path=args.test_names, **args.dataset_params)
    print('test size: ', len(data))
    model = load_model(args, data_sample=data[0], device=device)
    print('trainable params in model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    batch_size = args.batch_size if args.dataset_params['use_rec_atoms'] == False else 2
    collate_function = globals()[args.collate_function] if args.collate_params == {} else globals()[        #used for batch handling
        args.collate_function](**args.collate_params)
    loader = DataLoader(data, batch_size=batch_size, collate_fn=collate_function)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model.load_state_dict({k: v for k, v in checkpoint['model_state_dict'].items()})
    model.load_state_dict(checkpoint['model_state_dict'])   #should be path or loaded dic
    model.to(device)
    model.eval()

    for conformer_id in range(args.num_confs):
        all_h_feats_rec= []
        all_binding_sites = []
        all_batches = []
        all_gold_centers = []
        all_names = []
        data.conformer_id = conformer_id
        for batch in tqdm(enumerate(loader)):
            with torch.no_grad():
                 lig_graphs, rec_graphs, ligs_coords, recs_coords, all_rec_coords, pockets_coords_lig, geometry_graph, names, idx = tuple(
                     batch)
                
                 h_feats_rec, binding_sites, batches, gold_centers= model(lig_graphs,
                                                                        rec_graphs,
                                                                        complex_names=names,
                                                                        epoch=0,
                                                                        geometry_graph=geometry_graph.to(
                                                                            device) if geometry_graph != None else None)
                 for h_feats_rec, binding_sites, batches, gold_centers in zip(
                        h_feats_rec, ligs_coords, binding_sites, batches, gold_centers
                        ):
                    all_h_feats_rec.append(h_feats_rec.detach().cpu())
                    all_binding_sites.append(binding_sites.detach().cpu())
                    all_batches.append(gold_centers.detach().cpu())
                    all_gold_centers.append(gold_centers.detach().cpu())
                 all_names.extend(names)  
                     

        path = os.path.join(os.path.dirname(args.checkpoint),
                            f'predictions_Tune{tune_args != None}_RDKit{use_rdkit_coords}_confID{conformer_id}.pt')
        print(f'Saving predictions to {path}')
        results = {'embeddings': all_h_feats_rec, 'binding_site': all_binding_sites, 'batches': all_batches,
                   'lig_center': all_gold_centers,  'names': all_names}
        torch.save(results, path)
        
       




if __name__ == '__main__':
    args, cmdline_args = parse_arguments()

    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                if key in cmdline_args:
                    continue
                arg_dict[key] = value
        args.config = args.config.name
    else:
        config_dict = {}

    for run_dir in args.run_dirs:
        args.checkpoint = f'runs/{run_dir}/best_checkpoint.pt'
        config_dict['checkpoint'] = f'runs/{run_dir}/best_checkpoint.pt'
        # overwrite args with args from checkpoint except for the args that were contained in the config file
        arg_dict = args.__dict__
        with open(os.path.join(os.path.dirname(args.checkpoint), 'train_arguments.yaml'), 'r') as arg_file:
            checkpoint_dict = yaml.load(arg_file, Loader=yaml.FullLoader)
        for key, value in checkpoint_dict.items():
            if (key not in config_dict.keys()) and (key not in cmdline_args):
                if isinstance(value, list):
                    for v in value:
                        arg_dict[key].append(v)
                else:
                    arg_dict[key] = value
        #args.model_parameters['noise_initial'] = 0
        if args.inference_path == None:
            inference(args)
        
