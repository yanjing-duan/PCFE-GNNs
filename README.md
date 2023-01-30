# PCFE-GNNs
Pytorch implementation of “Improved GNNs for logD7.4 Prediction by Transferring Knowledge from Low-fidelity Data”

![image](https://user-images.githubusercontent.com/123799114/215254286-b348202c-5992-4f0b-aa61-4a62b13ec451.png)

* PCFE-GNNs means training graph neural networks (GNNs) using the PCFE strategy. PCFE works by pre-training a GNN model on 1.71 million computational logD data (low-fidelity data) and then fine-tuning it on 19,155 experimental logD7.4 data (high-fidelity data). The experiments for three GNN architectures (GCN, GAT, and Attentive FP) demonstrated the effectiveness of PCFE in improving GNNs for logD7.4 predictions. Moreover, the optimal PCFE-trained GNN model (cx-Attentive FP) achieves 0.909 of R² on the test set (1915 molecules) for logD7.4 prediction.

# Overview
* pretrain: contain the codes for pre-training the three GNNs (GCN, GAT, and Attentive FP).
* fine-tune: contain the codes for fine-tuning the three GNNs.
* descriptor-based_models: contain the codes for training four descriptor-based models (XGBoost, SVM, GB, and RF).
* data: data used for training models
* weights: contain the weights of the cx-Attentive FP model.
* predict: the code using the cx-Attentive FP model to predict logD7.4 for new chemicals and detailed information on how to obtain the predicted values are also provided in this file.



# Requirements
* python 3.6.12
* numpy 1.19.2
* pandas 1.1.3
* sklearn 0.20.3
* xgboost 1.0.2
* rdkit 2020.09.1.0
* openbabel 2.4.1
* torch 1.7.0
* DGL-LifeSci 0.2.6
