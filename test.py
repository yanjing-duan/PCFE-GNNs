import time
import pandas  as pd
from inference import SMLPredictor
import os
from rdkit import Chem
import openbabel as ob


models = ['GCN','GAT','AttentiveFP']
tasks = ['Aggregators','Promiscuous compounds','blue fluorescence','Artifacts','green fluorescence','Reactive compounds','Fluc inhibitors']
paths = [r'C:\Users\admin\Desktop\ChemFH_database_20201116\ChemFH_database_20201116\Summary_20201118\Summary_20201118.csv']


for path in paths:
    for model in models:
        for task in tasks:
            print(path,model,task)
            dir,name = os.path.split(path)
            name , suffix = name.split('.')
            df = pd.read_csv(path)
            smiles_list = df['mol'].tolist()

            predict_model = SMLPredictor(model=model,task=task)
            preds = predict_model.predict(smiles_list)
            df['predictions'] = preds.tolist()
            df.to_csv(dir+'/'+name+'_'+model+'_'+task+'.'+suffix)
