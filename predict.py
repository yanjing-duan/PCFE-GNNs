import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from datasets import Dataset
from dgllife.utils import EarlyStopping,RandomSplitter, MolecularWeightSplitter
from meter import Meter

from utils import set_random_seed, collate_molgraphs, load_model

'''
In order to predict logD7.4 values for new chemicals, you should follow the following steps:

First, fill in the SMILES string of the new chemical into the "Smiles" column of the "new_chemicals.csv" file.
Second, assign a number to the compound in the "index" column.
Finally, the prediction results will be saved to the "CX-AttentiveFP_prediction.csv" file in the "results" folder.
'''

def regress(args, model, bg):
    bg = bg.to(args['device'])
    if args['model'] == 'MPNN':
        h = bg.ndata['n_feat']
        e = bg.edata['e_feat']
        h, e = h.to(args['device']), e.to(args['device'])
        return model(bg, h, e)
    elif args['model'] in ['SchNet', 'MGCN']:
        node_types = bg.ndata['node_type']
        edge_distances = bg.edata['distance']
        node_types, edge_distances = node_types.to(args['device']), \
                                     edge_distances.to(args['device'])
        return model(bg, node_types, edge_distances)
    elif args['model'] in ['AttentiveFP']:
        atom_feats, bond_feats = bg.ndata['hv'], bg.edata['he']
        atom_feats, bond_feats = atom_feats.to(args['device']), bond_feats.to(args['device'])
        return model(bg, atom_feats, bond_feats)
    else:
        node_feats = bg.ndata[args['node_data_field']].to(args['device'])
        if args.get('edge_featurizer', None) is not None:
            edge_feats = bg.edata[args['edge_data_field']].to(args['device'])
            return model(bg, node_feats, edge_feats)
        else:
            return model(bg, node_feats)

def run_an_eval_epoch(args, model, data_loader):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, mask = batch_data
            labels = labels.to(args['device'])
            if bg.ndata == {}:
                continue
            prediction = regress(args, model, bg)
            eval_meter.update(prediction, labels)
        total_score = np.mean(eval_meter.compute_metric(args['metric_name']))
        mask, y_pred, y_true = eval_meter.finalize()
    return total_score, y_pred.numpy(), y_true.numpy()

def main(args):
    args['device'] = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_random_seed(args['random_seed'])

    # step 1: Read in new chemicals that the user wants to predict
    test_set = Dataset(path='new_chemicals.csv',smiles_to_graph=args['smiles_to_graph'],smiles_culumns=args['smiles_columns'],task_names=args['task_names'],
                                         node_featurizer=args.get('node_featurizer', None),
                                         edge_featurizer=args.get('edge_featurizer', None),load=False,
                      cache_file_path=f"data/logd-test-{args['model']}-data_dglgraph.bin")

    test_loader = DataLoader(dataset=test_set,
                             batch_size=args['batch_size'],
                             collate_fn=collate_molgraphs)

    model = load_model(args)
    model.to(args['device'])

    model.load_state_dict(torch.load("weights/AttentiveFP_chembl-CX_finetuning_weights_random.pth")['model_state_dict'],strict=False)

    _, test_y_pred, index = run_an_eval_epoch(args, model, test_loader)

    return test_y_pred, index

if __name__ == "__main__":
    print(torch.cuda.is_available())

    import argparse
    import os
    from configure import get_exp_configure

    parser = argparse.ArgumentParser(description='Regression model')
    parser.add_argument('-m', '--model', type=str,
                        choices=['GCN','GAT','Weave','AttentiveFP','MPNN', 'SchNet', 'MGCN'],
                        default='GCN',
                        help='Model to use')
    parser.add_argument('-mt', '--metric_name', type=str, default='R2',
                        help='Meter')

    args = parser.parse_args().__dict__

    args['model'] = 'AttentiveFP'

    args['smiles_columns'] = 'Smiles'
    args['task_names'] = ['index']

    args['exp'] = args['model']
    args.update(get_exp_configure(args['exp']))
    args['mode'] = 'higher'
    args['n_tasks'] = 1
    args['metric_name'] = 'R2'
    args['random_seed'] = 7
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    args['device'] = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    args['task'] = 'logd'

    if torch.cuda.is_available():
        print('GPU: ', os.environ["CUDA_VISIBLE_DEVICES"],
                  torch.cuda.get_device_properties(torch.cuda.current_device()))

    print(args['task'], '\t',args['model'], '\t', args['random_seed'])

    test_y_pred, index = main(args)

    results = pd.DataFrame.from_dict({"index": index.reshape(-1), "test_y_pred":test_y_pred.reshape(-1)}, orient='index')
    results = pd.DataFrame(results.values.T, index=results.columns, columns=results.index)

    print(results)

    # step 2: Save the prediction results to a user-specified csv file
    results.to_csv('results/CX-{}_prediction.csv'.format(args['model']), index=False)
