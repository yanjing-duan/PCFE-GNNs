import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from datasets import Dataset
from dgllife.utils import EarlyStopping,RandomSplitter, MolecularWeightSplitter
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

    train_set = Dataset(path='random_based_10/{}_train_set_random.csv'.format(args['num']),smiles_to_graph=args['smiles_to_graph'],smiles_culumns=args['smiles_columns'],task_names=args['task_names'],
                                         node_featurizer=args.get('node_featurizer', None),
                                         edge_featurizer=args.get('edge_featurizer', None),load=False,
                      cache_file_path=f"data/logd-train-{args['model']}-data_dglgraph.bin")

    test_set = Dataset(path='random_based_10/{}_test_set_random.csv'.format(args['num']),smiles_to_graph=args['smiles_to_graph'],smiles_culumns=args['smiles_columns'],task_names=args['task_names'],
                                         node_featurizer=args.get('node_featurizer', None),
                                         edge_featurizer=args.get('edge_featurizer', None),load=False,
                      cache_file_path=f"data/logd-test-{args['model']}-data_dglgraph.bin")

    val_set = Dataset(path='random_based_10/{}_dev_set_random.csv'.format(args['num']),smiles_to_graph=args['smiles_to_graph'],smiles_culumns=args['smiles_columns'],task_names=args['task_names'],
                                         node_featurizer=args.get('node_featurizer', None),
                                         edge_featurizer=args.get('edge_featurizer', None),load=False,
                      cache_file_path=f"data/logd-val-{args['model']}-data_dglgraph.bin")

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

    if args['pretraining']:
        model.load_state_dict(torch.load(args['pretraining_weights_path'])['model_state_dict'],strict=False)

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

    train_score, train_y_pred, train_y_true = run_an_eval_epoch(args, model, train_loader)
    val_score, val_y_pred, val_y_true = run_an_eval_epoch(args, model, val_loader)
    test_score,test_y_pred, test_y_true = run_an_eval_epoch(args, model, test_loader)


    test_mse = np.mean(np.square(test_y_true-test_y_pred))
    test_mae = np.mean(np.absolute(test_y_true-test_y_pred))
    val_mse = np.mean(np.square(val_y_true-val_y_pred))
    val_mae = np.mean(np.absolute(val_y_true-val_y_pred))
    train_mse = np.mean(np.square(train_y_true-train_y_pred))
    train_mae = np.mean(np.absolute(train_y_true-train_y_pred))

    print('train {} {:.4f}'.format(args['metric_name'], train_score))
    print('train mse {:.4f}'.format(train_mse))
    print('train mae {:.4f}'.format(train_mae))
    print('val {} {:.4f}'.format(args['metric_name'], val_score))
    print('val mse {:.4f}'.format(val_mse))
    print('val mae {:.4f}'.format(val_mae))
    print('test {} {:.4f}'.format(args['metric_name'], test_score))
    print('test mse {:.4f}'.format(test_mse))
    print('test mae {:.4f}'.format(test_mae))
    return train_score,train_mse,train_mae,test_score,test_mse,test_mae,val_score,val_mse,val_mae

if __name__ == "__main__":

    results = pd.DataFrame(columns=['train_score','train_mse','train_mae','test_score','test_mse','test_mae','val_score','val_mse','val_mae'],index=[1,2,3,4,5,6,7,8,9,10])
    for i in [1,2,3,4,5,6,7,8,9,10]:

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
        args['weights_path'] = 'weights/{}_chembl-CX_finetuning_weights_random.pth'.format(args['model'])

        args['pretraining'] = True # False、True
        args['pretraining_weights_path'] = 'weights/{}_chembl-CX_pretraining_weights.pth'.format(args['model'])
        args['num'] = i

        args['smiles_columns'] = 'Smiles'
        args['task_names'] = ['logD']

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

        train_score, train_mse, train_mae, test_score, test_mse, test_mae, val_score, val_mse, val_mae = main(args)
        print('best R2:',test_score)
        print('best mse:',test_mse)
        print('best mae:', test_mae)

        results.loc[i,'train_score'] = train_score
        results.loc[i, 'train_mse'] = train_mse
        results.loc[i, 'train_mae'] = train_mae
        results.loc[i, 'test_score'] = test_score
        results.loc[i, 'test_mse'] = test_mse
        results.loc[i, 'test_mae'] = test_mae
        results.loc[i, 'val_score'] = val_score
        results.loc[i, 'val_mse'] = val_mse
        results.loc[i, 'val_mae'] = val_mae

    print(results)

    if  args['pretraining'] :
        results.to_csv('results/{}_chembl-CX_finetuning_random.csv'.format(args['model']))
    else:
        results.to_csv('results/{}_no_pretraining_random.csv'.format(args['model']))