import numpy as np
import pandas as pd
import torch
import torch.nn as nn


from torch.utils.data import DataLoader

from smiles_dataset import SMILES_Dataset

from utils import collate_molgraphs_for_inference, load_model

def classify(args, model, bg):
    bg = bg.to(args['device'])
    node_feats = bg.ndata.pop(args['node_data_field']).to(args['device'])
    if args.get('edge_featurizer', None) is not None:
        edge_feats = bg.edata.pop(args['edge_data_field']).to(args['device'])
        return model(bg, node_feats, edge_feats)
    else:
        return model(bg, node_feats)



def run_an_eval_epoch(args, model, data_loader):
    model.eval()
    with torch.no_grad():

        prediction_list = []
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg = batch_data
            prediction = classify(args, model, bg)
            prediction_list.append(prediction.detach().cpu())

        y_pred = torch.sigmoid(torch.cat(prediction_list,dim=0)).numpy().reshape(-1)
    return y_pred


from configure import get_exp_configure
class SMLPredictor():
    def __init__(self,model='GAT',task='Aggregators'):
        # models = [ 'GAT','GCN','Weave','AttentiveFP', 'MPNN', 'SchNet', 'MGCN']
        # tasks = [r'FH_data_20200929\Aggregators.csv',r'FH_data_20200929\Blue fluorescence.csv',
        #          r'FH_data_20200929\FLuc inhibitors.csv',r'FH_data_20200929\Green fluorescence.csv',r'FH_data_20200929\Promiscuous compounds.csv',
        #          r'FH_data_20200929\Reactive compounds.csv','FH_data_20200929\Artifacts.csv','FH_data_20200929\Artifacts.csv']
        path = 'weight_file\{}.csv_{}_7.pth'.format(task,model)
        self.args = {}
        self.args['n_tasks'] = 1
        self.args['path'] = path
        self.args['model'] = model
        self.args['exp'] = self.args['model']
        self.args.update(get_exp_configure(self.args['exp']))

        self.args['device'] = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model = load_model(self.args)
        self.model.load_state_dict(torch.load(self.args['path'])['model_state_dict'])
        self.model.to(self.args['device'])

    def predict(self,sml_list):
        #['S(=O)(=O)(Oc1ccc(N=Nc2c(C)cc(-c3cc(C)c(NN=C4C(=O)C=Cc5c4c(S(=O)(=O)[O-])cc(S(=O)(=O)[O-])c5)cc3)cc2)cc1)c1ccc(C)cc1']
        dataset = SMILES_Dataset(sml_list=sml_list, smiles_to_graph=self.args['smiles_to_graph'],
                                 node_featurizer=self.args.get('node_featurizer', None),
                                 edge_featurizer=self.args.get('edge_featurizer', None))

        data_loader = DataLoader(dataset=dataset,
                                 batch_size=self.args['batch_size'],
                                 shuffle=False,
                                 collate_fn=collate_molgraphs_for_inference)

        y_pred = run_an_eval_epoch(self.args, self.model, data_loader)
        return y_pred
