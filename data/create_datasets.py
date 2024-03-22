import os
import argparse
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process .mat files and apply clustering.')
parser.add_argument('--files', nargs='+', help='List of .mat files containing order parameters')
parser.add_argument('--zeta_files', nargs='+', help='List of .mat files containing zeta parameters')
parser.add_argument('--dataset_name', type=str, help='Dataset name to be saved')
args = parser.parse_args()

# Use the command line arguments for files, zeta_files, and dataset_size
files = args.files if args.files is not None else []
zeta_files = args.zeta_files if args.zeta_files is not None else []
dataset_name = args.dataset_name

def validate_files(files, zeta_files):
    if not files:
        raise ValueError("No files provided. Please specify at least one file.")
    if not zeta_files:
        raise ValueError("No zeta files provided. Please specify at least one zeta file.")
    for file in files:
        if not os.path.isfile(file):
            raise FileNotFoundError(f"The file {file} does not exist.")
        if not file.endswith('.mat'):
            raise ValueError(f"The file {file} is not a .mat file.")
    for z_file in zeta_files:
        if not os.path.isfile(z_file):
            raise FileNotFoundError(f"The file {z_file} does not exist.")
        if not z_file.endswith('.mat'):
            raise ValueError(f"The file {z_file} is not a .mat file.")
    if len(files) != len(zeta_files):
        raise ValueError(f"You have entered {len(files)} file names but {len(zeta_files)} zeta files. Please try again after adding the missing files.")

if dataset_name is None:
    raise ValueError("No dataset_name provided. Please specify the dataset name as a string.")

LSI = []
q = []
Sk = []
zeta = []
Q6 = []
d5 = []

loaded_files = [loadmat(f) for f in files]
loaded_zeta_files = [loadmat(f) for f in zeta_files]

for i in range(len(loaded_files)):
    file = loaded_files[i]
    zeta_file = loaded_zeta_files[i]
    q_file = [item for sublist in file['q_all'] for item in sublist]
    LSI_file = [item for sublist in file['LSI_all'] for item in sublist]
    Sk_file = [item for sublist in file['Sk_all'] for item in sublist]
    Q6_file = [item for sublist in file['Q6_all'] for item in sublist]
    d5_file = [item for sublist in file['d5_all'] for item in sublist]
    z_file = [item for sublist in zeta_file['zeta_all'] for item in sublist]
    q.extend(q_file)
    LSI.extend(LSI_file)
    Sk.extend(Sk_file)
    zeta.extend(z_file)
    Q6.extend(Q6_file)
    d5.extend(d5_file)

def scale(df):
    x = df.values
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_scaled = pd.DataFrame(x_scaled, columns=df.columns)
    return df_scaled

data = {
        'Sk_all': Sk,
        'LSI_all': LSI,
        'zeta_all': zeta,
        'd5_all': d5,
        'q_all': q,
        'Q6_all': Q6
       }

df = pd.DataFrame(data=data)
df_scaled = scale(df)

df.to_csv(f"{dataset_name}_unscaled.csv", index=False)
df_scaled.to_csv(f"{dataset_name}_scaled.csv", index=False)
