# A Tale of Two States: Investigating Water's Mixture-like Properties using Machine Learning Methods

All the code used to perform the analysis is included in this repository. Unfortunately, the datasets: both input (order parameters) and output (clustering results) are available on request. In order to run the notebooks, you will need the input files. After obtaining the input files on request, place them in the `data` folder and then run `data_processing.sh`. It will perform the necessary combining and scaling of the files to generate the scaled and unscaled datasets. 

For computationally heavy scripts that were not run as notebooks, the `.py` files have been included in this repo. Their output (generated after running on AWS Sagemaker instances) has been provided in the `output` folder.
