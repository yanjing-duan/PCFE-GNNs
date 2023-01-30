import dgl
import numpy as np
import random
import torch

from dgllife.utils import one_hot_encoding
from dgllife.utils.splitters import RandomSplitter
from functools import partial

def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use. Default to 0.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def chirality(atom):
    """Get Chirality information for an atom.
    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    Returns
    -------
    list of 3 boolean values
        The 3 boolean values separately indicate whether the atom
        has a chiral tag R, whether the atom has a chiral tag S and
        whether the atom is a possible chiral center.
    """
    try:
        return one_hot_encoding(atom.GetProp('_CIPCode'), ['R', 'S']) + \
               [atom.HasProp('_ChiralityPossible')]
    except:
        return [False, False] + [atom.HasProp('_ChiralityPossible')]

def load_dataset_for_classification(args):
    """Load dataset for classification tasks.
    Parameters
    ----------
    args : dict
        Configurations.
    Returns
    -------
    dataset
        The whole dataset.
    train_set
        Subset for training.
    val_set
        Subset for validation.
    test_set
        Subset for test.
    """
    assert args['dataset'] in ['Tox21']
    if args['dataset'] == 'Tox21':
        from dgllife.data import Tox21
        dataset = Tox21(smiles_to_graph=partial(args['smiles_to_graph'], add_self_loop=True),
                        node_featurizer=args.get('node_featurizer', None),
                        edge_featurizer=args.get('edge_featurizer', None),
                        load=False,
                        cache_file_path='./' + args['exp'])
        train_set, val_set, test_set = RandomSplitter.train_val_test_split(
            dataset, frac_train=args['frac_train'], frac_val=args['frac_val'],
            frac_test=args['frac_test'], random_state=args['random_seed'])

    return dataset, train_set, val_set, test_set

def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader.
    Parameters
    ----------
    data : list of 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and a binary
        mask indicating the existence of labels.
    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels.
    """
    assert len(data[0]) in [3, 4], \
        'Expect the tuple to be of length 3 or 4, got {:d}'.format(len(data[0]))
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
        masks = None
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))

    from copy import deepcopy
    graphs = deepcopy(graphs)
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)
    if len(labels.shape)==1:
        labels = labels.unsqueeze(1)
    masks = torch.stack(masks, dim=0)
    if (masks is not None) and (len(masks.shape)==1):
        masks = masks.unsqueeze(1)
    return smiles, bg, labels, masks



def collate_molgraphs_for_inference(data):
    """Batching a list of datapoints for dataloader.
    Parameters
    ----------
    data : list of 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and a binary
        mask indicating the existence of labels.
    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels.
    """
    assert len(data[0]) in [2], \
        'Expect the tuple to be of length 1, got {:d}'.format(len(data[0]))

    smiles, graphs = map(list, zip(*data))

    from copy import deepcopy
    graphs = deepcopy(graphs)
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)

    return smiles, bg

def load_model(args):
    if args['model'] == 'AttentiveFP':
        from dgllife.model import AttentiveFPPredictor
        model = AttentiveFPPredictor(node_feat_size=args['node_featurizer'].feat_size('hv'),
                                     edge_feat_size=args['edge_featurizer'].feat_size('he'),
                                     num_layers=args['num_layers'],
                                     num_timesteps=args['num_timesteps'],
                                     graph_feat_size=args['graph_feat_size'],
                                     n_tasks=args['n_tasks'],
                                     dropout=args['dropout'])
    if args['model'] == 'GCN':
        from gcn_predictor import GCNPredictor
        model = GCNPredictor(in_feats=args['node_featurizer'].feat_size(),
                             hidden_feats=args['gcn_hidden_feats'],
                             predictor_hidden_feats=args['predictor_hidden_feats'],
                             n_tasks=args['n_tasks'],predictor_dropout=args['dropout'])

    if args['model'] == 'GAT':
        from gat_predcitor import GATPredictor
        model = GATPredictor(in_feats=args['node_featurizer'].feat_size(),
                             hidden_feats=args['gat_hidden_feats'],
                             num_heads=args['num_heads'],
                             predictor_hidden_feats=args['predictor_hidden_feats'],
                             n_tasks=args['n_tasks'],predictor_dropout=args['dropout'])

    if args['model'] == 'Weave':
        from dgllife.model import WeavePredictor
        model = WeavePredictor(node_in_feats=args['node_featurizer'].feat_size(),
                               edge_in_feats=args['edge_featurizer'].feat_size(),
                               num_gnn_layers=args['num_gnn_layers'],
                               gnn_hidden_feats=args['gnn_hidden_feats'],
                               graph_feats=args['graph_feats'],
                               n_tasks=args['n_tasks'])
    if args['model'] == 'SchNet':
        from dgllife.model import SchNetPredictor
        model = SchNetPredictor(node_feats=args['node_feats'],
                                hidden_feats=args['hidden_feats'],
                                predictor_hidden_feats=args['predictor_hidden_feats'],
                                n_tasks=args['n_tasks'])

    if args['model'] == 'MGCN':
        from dgllife.model import MGCNPredictor
        model = MGCNPredictor(feats=args['feats'],
                              n_layers=args['n_layers'],
                              predictor_hidden_feats=args['predictor_hidden_feats'],
                              n_tasks=args['n_tasks'])

    if args['model'] == 'MPNN':
        from dgllife.model import MPNNPredictor
        model = MPNNPredictor(node_in_feats=args['node_in_feats'],
                              edge_in_feats=args['edge_in_feats'],
                              node_out_feats=args['node_out_feats'],
                              edge_hidden_feats=args['edge_hidden_feats'],
                              n_tasks=args['n_tasks'])

    return model