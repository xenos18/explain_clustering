# Clustering Results Explainer

![Logo](logo.png)

> A short description of the project.

## Table of Contents

- [Objective](#objective) - [Installation](#installation) - [Usage](#usage) - [Documentation](#documentation) - [License](#license)

## Objective

In this project, we developed an algorithm that analyzes the results of a clustering algorithm. The objective of this project is to simplify the analysis of the clustering results and provide insights to data analysts and researchers.

## Installation

To install the clustering result analyzer, follow these steps:

1. Clone the repository or download the zip file. 

2. Install the required dependencies using `pip install -r requirements.txt`. 

3. Run the algorithm with the command `python main.py`.

## Usage

The clustering result analyzer is easy to use. Simply provide the output file of a clustering algorithm and the algorithm will analyze the results and generate insights. You can also specify various parameters to customize the analysis process.

```
from sklearn import datasets
from sklearn.clusters import Kmeans
from explainer import BayesExplainer

#Data Loading
df = datasets.make_blobs(n_samples=n_samples, random_state=8)
features = df.columns

#Clustering
kmeans = KMeans(init="k-means++", n_clusters=5,n_init=10, max_iter=1000, random_state=24,)
kmeans.fit(df)
df['k_means_labels'] = kmeans.labels_

#Interpretation
bi_explainer = BayesExplainer(data=df, label_col='k_means_labels', cat_features=None, num_features=features)
significance_matrix = bi_explainer.explain(verbose=False)

```

## Documentation

The clustering result analyzer consists of the following files:

- `main.py`: The main implementation of the algorithm. - `utils.py`: Utility functions to analyze the clustering results. - `clustering_output.csv`: A sample clustering output to demonstrate the usage of the algorithm. - `requirements.txt`: A file containing the required dependencies.

The clustering result analyzer analyzes the clustering output using various techniques such as visualization, statistical analysis, and clustering evaluation metrics. It provides insights such as the number of clusters, the distribution of data points, and the quality of the clustering.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use the clustering algorithm or clustering result analyzer in your research or work, please cite the following paper:

Balabaeva, K., & Kovalchuk, S. (2020). Post-hoc interpretation of clinical pathways clustering using Bayesian inference. Procedia Computer Science, 178, 264-273.

``` @article{balabaeva2020post, title={Post-hoc interpretation of clinical pathways clustering using Bayesian inference}, author={Balabaeva, K. and Kovalchuk, S.}, journal={Procedia Computer Science}, volume={178}, pages={264--273}, year={2020}, publisher={Elsevier} } ```

Thank you for your support!
