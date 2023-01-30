# Creating datasets from .csv files for molecular property prediction.

import dgl.backend as F
import numpy as np
import os

from dgl.data.utils import save_graphs, load_graphs

__all__ = ['SMILES_Dataset']

class SMILES_Dataset(object):
    """MoleculeCSVDataset

    This is a general class for loading molecular data from :class:`pandas.DataFrame`.

    In data pre-processing, we construct a binary mask indicating the existence of labels.

    All molecules are converted into DGLGraphs. After the first-time construction, the
    DGLGraphs can be saved for reloading so that we do not need to reconstruct them every time.

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe including smiles and labels. Can be loaded by pandas.read_csv(file_path).
        One column includes smiles and some other columns include labels.
    smiles_to_graph: callable, str -> DGLGraph
        A function turning a SMILES string into a DGLGraph.
    node_featurizer : None or callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule, which can be used to update
        ndata for a DGLGraph.
    edge_featurizer : None or callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for edges like bonds in a molecule, which can be used to update
        edata for a DGLGraph.
    smiles_column: str
        Column name for smiles in ``df``.
    cache_file_path: str
        Path to store the preprocessed DGLGraphs. For example, this can be ``'dglgraph.bin'``.
    task_names : list of str or None
        Columns in the data frame corresponding to real-valued labels. If None, we assume
        all columns except the smiles_column are labels. Default to None.
    load : bool
        Whether to load the previously pre-processed dataset or pre-process from scratch.
        ``load`` should be False when we want to try different graph construction and
        featurization methods and need to preprocess from scratch. Default to True.
    log_every : bool
        Print a message every time ``log_every`` molecules are processed. Default to 1000.
    init_mask : bool
        Whether to initialize a binary mask indicating the existence of labels. Default to True.
    """
    def __init__(self, sml_list, smiles_to_graph, node_featurizer, edge_featurizer,
                log_every=10000):

        self.smiles = sml_list

        self._pre_process(smiles_to_graph, node_featurizer, edge_featurizer,
                           log_every)

    def _pre_process(self, smiles_to_graph, node_featurizer,
                     edge_featurizer,  log_every):
        """Pre-process the dataset

        * Convert molecules from smiles format into DGLGraphs
          and featurize their atoms
        * Set missing labels to be 0 and use a binary masking
          matrix to mask them

        Parameters
        ----------
        smiles_to_graph : callable, SMILES -> DGLGraph
            Function for converting a SMILES (str) into a DGLGraph.
        node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
            Featurization for nodes like atoms in a molecule, which can be used to update
            ndata for a DGLGraph.
        edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
            Featurization for edges like bonds in a molecule, which can be used to update
            edata for a DGLGraph.
        load : bool
            Whether to load the previously pre-processed dataset or pre-process from scratch.
            ``load`` should be False when we want to try different graph construction and
            featurization methods and need to preprocess from scratch. Default to True.
        log_every : bool
            Print a message every time ``log_every`` molecules are processed.
        init_mask : bool
            Whether to initialize a binary mask indicating the existence of labels.
        """
        print('Processing dgl graphs from scratch...')
        self.graphs = []

        for i, s in enumerate(self.smiles):
            if (i + 1) % log_every == 0:
                print('Processing molecule {:d}/{:d}'.format(i+1, len(self)))

            graph = smiles_to_graph(s, node_featurizer=node_featurizer,
                            edge_featurizer=edge_featurizer)

            self.graphs.append(graph)


    def __getitem__(self, item):
        """Get datapoint with index

        Parameters
        ----------
        item : int
            Datapoint index

        Returns
        -------
        str
            SMILES for the ith datapoint
        DGLGraph
            DGLGraph for the ith datapoint
        Tensor of dtype float32 and shape (T)
            Labels of the datapoint for all tasks
        Tensor of dtype float32 and shape (T), optional
            Binary masks indicating the existence of labels for all tasks. This is only
            generated when ``init_mask`` is True in the initialization.
        """

        return self.smiles[item], self.graphs[item]

    def __len__(self):
        """Size for the dataset

        Returns
        -------
        int
            Size for the dataset
        """
        return len(self.smiles)
