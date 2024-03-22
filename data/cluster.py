import argparse
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('dataset_name', type=str, help='Name of the dataset')
args = parser.parse_args()

# Read the relevant datasets
dataset_name = args.dataset_name
df = pd.read_csv(f"{dataset_name}_unscaled.csv")
df_scaled = pd.read_csv(f"{dataset_name}_scaled.csv")

# Fit a BGMM on the dataset using all the features 
features = ['LSI_all', 'zeta_all', 'd5_all', 'Sk_all', 'q_all', 'Q6_all']
gmm = BayesianGaussianMixture(covariance_type="full",
                              n_components=2,
                              tol=1e-3,
                              max_iter=200,
                              mean_precision_prior=None,
                              weight_concentration_prior=None)
gmm.fit(df_scaled[features])
labels = gmm.predict(df_scaled[features])

# Adjust labels so 0 is the larger cluster
zero_count = (labels == 0).sum()
one_count = (labels == 1).sum()
if zero_count < one_count:
    labels = 1 - labels  # This flips 0s to 1s and 1s to 0s
df['labels'] = labels

df.to_csv(f"{dataset_name}_clustered.csv", index=False)
