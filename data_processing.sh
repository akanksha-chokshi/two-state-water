#!/bin/bash

# Navigate to the data directory
cd data

# Create datasets
python3 create_datasets.py --files OrderParam_Run21_swm4ndp_T30.0.mat OrderParam_Run22_swm4ndp_T30.0.mat OrderParam_Run23_swm4ndp_T30.0.mat --zeta_files OrderParamZeta_Run21_swm4ndp_T30.0.mat OrderParamZeta_Run22_swm4ndp_T30.0.mat OrderParamZeta_Run23_swm4ndp_T30.0.mat --dataset_name 1024
python3 create_datasets.py --files OrderParam_Run31_swm4ndp_T30.0.mat OrderParam_Run32_swm4ndp_T30.0.mat OrderParam_Run33_swm4ndp_T30.0.mat --zeta_files OrderParamZeta_Run31_swm4ndp_T30.0.mat OrderParamZeta_Run32_swm4ndp_T30.0.mat OrderParamZeta_Run33_swm4ndp_T30.0.mat --dataset_name 768
python3 create_datasets.py --files OrderParam_Run11_swm4ndp_T30.0.mat OrderParam_Run12_swm4ndp_T30.0.mat OrderParam_Run13_swm4ndp_T30.0.mat --zeta_files OrderParamZeta_Run11_swm4ndp_T30.0.mat OrderParamZeta_Run12_swm4ndp_T30.0.mat OrderParamZeta_Run13_swm4ndp_T30.0.mat --dataset_name 512

# Run clustering
python3 cluster.py 1024
python3 cluster.py 768
python3 cluster.py 512
