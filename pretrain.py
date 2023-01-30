import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from datasets import Dataset
from dgllife.utils import EarlyStopping,RandomSplitter, MolecularWeightSplitter,ScaffoldSplitter
from meter import Meter

from utils import set_random_seed, collate_molgraphs, load_model

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

def run_a_train_epoch(args, epoch, model, data_loader,
                      loss_criterion, optimizer):
    model.train()
    train_meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels,mask = batch_data
        labels = labels.to(args['device'])
        prediction = regress(args, model, bg)
        loss = (loss_criterion(prediction, labels)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(prediction, labels)
    total_score = np.mean(train_meter.compute_metric(args['metric_name']))
    print('epoch {:d}/{:d}, training {} {:.4f}'.format(
        epoch + 1, args['num_epochs'], args['metric_name'], total_score))

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

    dataset = Dataset(path=args['task'],smiles_to_graph=args['smiles_to_graph'],smiles_culumns=args['smiles_columns'],task_names=args['task_names'],
                                         node_featurizer=args.get('node_featurizer', None),
                                         edge_featurizer=args.get('edge_featurizer', None),
                      cache_file_path=f"data/{args['task'].split('.')[0]}-{args['model']}-data_dglgraph.bin")

    train_set, val_set, test_set = RandomSplitter.train_val_test_split(
        dataset, frac_train=args['frac_train'], frac_val=args['frac_val'],
        frac_test=args['frac_test'])
    train_loader = DataLoader(dataset=train_set,
                              batch_size=args['batch_size'],
                              shuffle=True,
                              collate_fn=collate_molgraphs)
    val_loader = DataLoader(dataset=val_set,
                            batch_size=args['batch_size'],
                            collate_fn=collate_molgraphs)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=args['batch_size'],
                             collate_fn=collate_molgraphs)

    model = load_model(args)
    loss_fn = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                                 weight_decay=args['weight_decay'])
    stopper = EarlyStopping(mode=args['mode'], patience=args['patience'],filename=args['weights_path'])
    model.to(args['device'])

    for epoch in range(args['num_epochs']):
        # Train
        run_a_train_epoch(args, epoch, model, train_loader, loss_fn, optimizer)

        # Validation and early stop
        val_score,_,_ = run_an_eval_epoch(args, model, val_loader)
        early_stop = stopper.step(val_score, model)
        print('epoch {:d}/{:d}, validation {} {:.4f}, best validation {} {:.4f}'.format(
            epoch + 1, args['num_epochs'], args['metric_name'], val_score,
            args['metric_name'], stopper.best_score))

        if early_stop:
            break

    stopper.load_checkpoint(model)
    test_score,y_pred, y_true = run_an_eval_epoch(args, model, test_loader)
    mse = np.mean(np.square(y_true-y_pred))
    mae = np.mean(np.absolute(y_true-y_pred))

    print('test {} {:.4f}'.format(args['metric_name'], test_score))
    print('test mse {:.4f}'.format(mse))
    print('test mae {:.4f}'.format(mae))

    return test_score, mse,mae

if __name__ == "__main__":
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
    parser.add_argument('-t', '--task', type=str, default='caco2',
                        help='data path')
    args = parser.parse_args().__dict__

    args['task'] = r'chembl_CXlogD_1711398.csv'

    args['model'] = 'AttentiveFP'
    args['weights_path'] = 'weights/AttentiveFP_chembl-CX_pretraining_weights.pth'

    args['smiles_columns'] = 'Smiles'
    args['task_names'] = ['CX LogD']

    args['exp'] = args['model']
    args.update(get_exp_configure(args['exp']))
    args['mode'] = 'higher'
    args['n_tasks'] = 1
    args['metric_name'] = 'R2'
    args['random_seed'] = 7

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    args['device'] = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if torch.cuda.is_available():
        print('GPU: ', os.environ["CUDA_VISIBLE_DEVICES"],
                  torch.cuda.get_device_properties(torch.cuda.current_device()))

    print(args['task'], '\t',args['model'], '\t', args['random_seed'])

    test_score, mse_score, mae_score = main(args)
    print('best R2:',test_score)
    print('best mse:',mse_score)
    print('best mae:',mae_score)