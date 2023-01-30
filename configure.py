# graph construction

from mol_to_graph import smiles_to_bigraph, smiles_to_complete_graph
# general featurization
from dgllife.utils import ConcatFeaturizer
# node featurization
from dgllife.utils import BaseAtomFeaturizer, atom_type_one_hot, atom_degree_one_hot, \
    atom_formal_charge, atom_num_radical_electrons, \
    atom_hybridization_one_hot, atom_total_num_H_one_hot
# edge featurization
from dgllife.utils.featurizers import BaseBondFeaturizer,atom_type_one_hot, atom_hybridization_one_hot, atom_is_aromatic
from dgllife.utils import CanonicalAtomFeaturizer, WeaveAtomFeaturizer,AttentiveFPAtomFeaturizer,AttentiveFPBondFeaturizer
# edge featurization
from dgllife.utils import WeaveEdgeFeaturizer
from functools import partial
import numpy as np

from utils import chirality
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import os.path as osp
from dgl import backend as F

def alchemy_nodes(mol):
    """Featurization for all atoms in a molecule. The atom indices
    will be preserved.
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object
    Returns
    -------
    atom_feats_dict : dict
        Dictionary for atom features
    """
    atom_feats_dict = defaultdict(list)
    is_donor = defaultdict(int)
    is_acceptor = defaultdict(int)

    fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    mol_featurizer = ChemicalFeatures.BuildFeatureFactory(fdef_name)
    mol_feats = mol_featurizer.GetFeaturesForMol(mol)
    # mol_conformers = mol.GetConformers()
    # assert len(mol_conformers) == 1

    for i in range(len(mol_feats)):
        if mol_feats[i].GetFamily() == 'Donor':
            node_list = mol_feats[i].GetAtomIds()
            for u in node_list:
                is_donor[u] = 1
        elif mol_feats[i].GetFamily() == 'Acceptor':
            node_list = mol_feats[i].GetAtomIds()
            for u in node_list:
                is_acceptor[u] = 1

    num_atoms = mol.GetNumAtoms()
    for u in range(num_atoms):
        atom = mol.GetAtomWithIdx(u)
        atom_type = atom.GetAtomicNum()
        num_h = atom.GetTotalNumHs()
        atom_feats_dict['node_type'].append(atom_type)

        h_u = []
        h_u += atom_type_one_hot(atom, ['H', 'C', 'N', 'O', 'F', 'S', 'Cl'])
        h_u.append(atom_type)
        h_u.append(is_acceptor[u])
        h_u.append(is_donor[u])
        h_u += atom_is_aromatic(atom)
        h_u += atom_hybridization_one_hot(atom, [Chem.rdchem.HybridizationType.SP,
                                                 Chem.rdchem.HybridizationType.SP2,
                                                 Chem.rdchem.HybridizationType.SP3])
        h_u.append(num_h)
        atom_feats_dict['n_feat'].append(F.tensor(np.array(h_u).astype(np.float32)))

    atom_feats_dict['n_feat'] = F.stack(atom_feats_dict['n_feat'], dim=0)
    atom_feats_dict['node_type'] = F.tensor(np.array(
        atom_feats_dict['node_type']).astype(np.int64))

    return atom_feats_dict

def alchemy_edges(mol, self_loop=False):
    """Featurization for all bonds in a molecule.
    The bond indices will be preserved.
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object
    self_loop : bool
        Whether to add self loops. Default to be False.
    Returns
    -------
    bond_feats_dict : dict
        Dictionary for bond features
    """
    bond_feats_dict = defaultdict(list)


    AllChem.EmbedMolecule(mol)
    mol_conformers = mol.GetConformers()

    if len(mol_conformers)>0:
        geom = mol_conformers[0].GetPositions()
    else:
        return None

    num_atoms = mol.GetNumAtoms()
    for u in range(num_atoms):
        for v in range(num_atoms):
            if u == v and not self_loop:
                continue

            e_uv = mol.GetBondBetweenAtoms(u, v)
            if e_uv is None:
                bond_type = None
            else:
                bond_type = e_uv.GetBondType()
            bond_feats_dict['e_feat'].append([
                float(bond_type == x)
                for x in (Chem.rdchem.BondType.SINGLE,
                          Chem.rdchem.BondType.DOUBLE,
                          Chem.rdchem.BondType.TRIPLE,
                          Chem.rdchem.BondType.AROMATIC, None)
            ])
            bond_feats_dict['distance'].append(
                np.linalg.norm(geom[u] - geom[v]))

    bond_feats_dict['e_feat'] = F.tensor(
        np.array(bond_feats_dict['e_feat']).astype(np.float32))
    bond_feats_dict['distance'] = F.tensor(
        np.array(bond_feats_dict['distance']).astype(np.float32)).reshape(-1 , 1)

    return bond_feats_dict

GCN = {
    'random_seed': 2,
    'batch_size': 128,
    'lr': 10**-2.5,
    'num_epochs': 100,
    'node_data_field': 'h',
    'frac_train': 0.8,
    'frac_val': 0.1,
    'frac_test': 0.1,
    'in_feats':AttentiveFPAtomFeaturizer().feat_size(),
    'gcn_hidden_feats': [256,256],
    'predictor_hidden_feats': 128,
    'patience': 10,
    'smiles_to_graph': partial(smiles_to_bigraph, add_self_loop=True),
    'node_featurizer': AttentiveFPAtomFeaturizer(),
    'metric_name': 'roc_auc_score',
    'mode':'higher',
    'weight_decay': 10**-8,
    'dropout': 0
}

GAT = {
    'random_seed': 2,
    'batch_size': 128,
    'lr': 10**-3.5,
    'num_epochs': 100,
    'node_data_field': 'h',
    'frac_train': 0.8,
    'frac_val': 0.1,
    'frac_test': 0.1,
    'in_feats': AttentiveFPAtomFeaturizer().feat_size(),
    'gat_hidden_feats': [128,128,128],
    'predictor_hidden_feats': 256,
    'num_heads': [4, 4, 4],
    'patience': 10,
    'smiles_to_graph': partial(smiles_to_bigraph, add_self_loop=True),
    'node_featurizer': AttentiveFPAtomFeaturizer(),
    'metric_name': 'roc_auc_score',
    'mode': 'higher',
    'weight_decay': 0,
    'dropout':0.0,

}

Weave = {
    'random_seed': 2,
    'batch_size': 128,
    'lr': 1e-3,
    'num_epochs': 100,
    'node_data_field': 'h',
    'edge_data_field': 'e',
    'frac_train': 0.8,
    'frac_val': 0.1,
    'frac_test': 0.1,
    'weight_decay': 0,
    'num_gnn_layers': 2,
    'gnn_hidden_feats': 50,
    'graph_feats': 128,
    'patience': 10,
    'smiles_to_graph': partial(smiles_to_complete_graph, add_self_loop=True),
    'node_featurizer': WeaveAtomFeaturizer(),
    'edge_featurizer': WeaveEdgeFeaturizer(max_distance=2),
    'metric_name': 'roc_auc_score',
    'mode':'higher'
}

attentivefp = {
    'random_seed': 8,
    'graph_feat_size': 256,
    'num_layers': 3,
    'num_timesteps': 1,
    'node_feat_size':  AttentiveFPAtomFeaturizer().feat_size(),
    'edge_feat_size': AttentiveFPBondFeaturizer().feat_size(),
    'n_tasks': 1,
    'dropout': 0.1,
    'weight_decay': 10 ** (-6.0),
    'lr': 10**-3.5,
    'batch_size': 128,
    'num_epochs': 100,
    'frac_train': 0.8,
    'frac_val': 0.1,
    'frac_test': 0.1,
    'patience': 20,
    'metric_name': 'rmse',
    'mode': 'higher',
    'node_data_field': 'hv',
    'edge_data_field': 'he',
    'smiles_to_graph': smiles_to_bigraph,
    # Follow the atom featurization in the original work
    'node_featurizer': AttentiveFPAtomFeaturizer(atom_data_field='hv'),
    'edge_featurizer': AttentiveFPBondFeaturizer(bond_data_field='he'),
}

MPNN = {
    'node_data_field': 'n_feat',
    'edge_data_field': 'e_feat',
    'random_seed': 0,
    'batch_size': 32,
    'num_epochs': 100,
    'node_in_feats': 15,
    'node_out_feats': 64,
    'edge_in_feats': 5,
    'edge_hidden_feats': 32,
    'n_tasks': 12,
    'lr': 10**-1.5,
    'patience': 10,
    'metric_name': 'mae',
    'frac_train': 0.8,
    'frac_val': 0.1,
    'frac_test': 0.1,
    'mode': 'higher',
    'weight_decay':0 ,
    'smiles_to_graph': smiles_to_complete_graph,
    'node_featurizer': alchemy_nodes,
    'edge_featurizer': alchemy_edges,
}

SchNet = {
    'node_data_field': 'node_type',
    'edge_data_field': 'distance',
    'random_seed': 0,
    'batch_size': 64,
    'num_epochs': 100,
    'node_feats': 64,
    'hidden_feats': [64, 64, 64],
    'predictor_hidden_feats': 64,
    'n_tasks': 12,
    'lr': 0.001,
    'patience': 10,
    'metric_name': 'mae',
    'frac_train': 0.8,
    'frac_val': 0.1,
    'frac_test': 0.1,
    'mode': 'higher',
    'weight_decay': 0,
    'smiles_to_graph': smiles_to_complete_graph,
    'node_featurizer': alchemy_nodes,
    'edge_featurizer': alchemy_edges,
}

MGCN = {
    'node_data_field': 'node_type',
    'edge_data_field': 'distance',
    'random_seed': 0,
    'batch_size': 64,
    'num_epochs': 100,
    'feats': 128,
    'n_layers': 3,
    'predictor_hidden_feats': 64,
    'n_tasks': 12,
    'lr': 0.001,
    'frac_train': 0.8,
    'frac_val': 0.1,
    'frac_test': 0.1,
    'patience': 10,
    'metric_name': 'mae',
    'mode': 'higher',
    'weight_decay': 0,
    'smiles_to_graph': smiles_to_complete_graph,
    'node_featurizer': alchemy_nodes,
    'edge_featurizer': alchemy_edges,
}


experiment_configures = {
    'AttentiveFP': attentivefp,
    'GCN': GCN,
    'GAT': GAT,
    'Weave': Weave,
    'MPNN': MPNN,
    'SchNet': SchNet,
    'MGCN': MGCN
}
def get_exp_configure(exp_name):
    return experiment_configures[exp_name]