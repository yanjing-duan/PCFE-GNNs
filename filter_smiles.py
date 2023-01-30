import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import multiprocessing as mp
import numpy as np
import os




def validate(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        AllChem.EmbedMolecule(mol)
        mol_conformers = mol.GetConformers()
        if len(mol_conformers) >= 1:
             return True
        else:
            return None
    except:
        return None
def tasks(df,i):
    length1 = len(df)
    length2 = int(np.ceil(length1/32))+1
    start = i*length2
    end  = (i+1)*length2

    df = df.iloc[start:end,:].copy()
    df['is_valid'] = df['SMILES'].map(validate)
    print(len(df))
    df = df.dropna(axis=0)
    return df


if __name__=='__main__':
    path = 'FH_data_20200929'
    list_dirs = os.listdir(path)
    list_dirs = [os.path.join(path,name) for name in list_dirs]
    # list_dirs = ['FH_data_20200929\Promiscuous compounds.csv']
    print(list_dirs)
    for dir in list_dirs:
        # dir = 'FH_data_20200929\Blue fluorescence.csv'
        if not dir.endswith('.csv'):
            continue
        print(dir)
        df = pd.read_csv(dir)
        df = df.dropna(axis=0)
        df = df[['SMILES','Label']]
        length1 = len(df)
        pool = mp.Pool(32)
        cbs = []

        cbs = pool.starmap_async(tasks,[(df,i) for i in range(32)])
        pool.close()
        pool.join()


        results = cbs.get()
        df = pd.concat(results,axis=0)
        # df['is_valid'] = df['SMILES'].map(validate)

        df = df.dropna(axis=0)
        length2 = len(df)

        df.to_csv(dir,index=False)

        print(length2 - length1)