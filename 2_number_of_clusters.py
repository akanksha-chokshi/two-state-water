import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

df = pd.read_csv("data/1024_scaled.csv").dropna()

# Fit a BGMM on the dataset using all the features 
features = ['LSI_all', 'zeta_all', 'd5_all', 'Sk_all', 'q_all', 'Q6_all']
range_n_clusters = [2, 3, 4, 5]


# Open a text file in write mode
with open('output/num_clusters_output.txt', 'w') as file:
    # Write the results to the file
    for n_clusters in range_n_clusters:
        bgmm = BayesianGaussianMixture(covariance_type="full", n_components=n_clusters)
        cluster_labels = bgmm.fit_predict(df[features])
        silhouette_avg = silhouette_score(df[features], cluster_labels)
        print(f"For n_clusters={n_clusters}, the average silhouette_score is: {silhouette_avg}")
        ch_avg = calinski_harabasz_score(df[features], cluster_labels)
        print(f"For n_clusters={n_clusters}, the average calinski_harabasz_score is: {ch_avg}")
        db_avg = davies_bouldin_score(df[features], cluster_labels)
        print(f"For n_clusters={n_clusters}, the average davies_bouldin_score is: {db_avg}")

