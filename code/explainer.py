import numpy as np
import matplotlib.pyplot as plt
#TO DO: модуль для определение категориальных признаков


class BayesExplainer():

    def __init__(self, data, label_col, cat_features, num_features):
        #определение кластерной матрицы
        self.data = data
        self.label_col = label_col
        self.cat_features = cat_features
        self.num_features = num_features
        self.num_observations = self.data[label_col].value_counts()
        self.cluster_table = self._create_cluster_table()
        
    
    def _create_cluster_mtrx(self):
        #TO DO numerical case
        cluster_table = self.data[self.cat_features+self.num_features+[self.label_col]].groupby([self.label_col]).sum()
        cluster_table['num_observations'] = self.num_observations
        cluster_table['cluster id'] = cluster_table.index
        return cluster_table
        

    def calc_feature_prob_in_cluster(self):
        for column in self.cat_features:
            self.cluster_table[column + ' prob'] = self.cluster_table[[column, 'num_observations']].apply(lambda a: a[0]/a[1], axis=1)

    def explain(self):
        self.calc_feature_prob_in_cluster()
        self.plot_feature_prob_in_clusters()


    def plot_feature_prob_in_cluster(self, column):
        '''
        Plots hist plot for feature probability across clusters
        '''
        column = column + ' prob'
        names = list(self.cluster_table.index)
        values = list(self.cluster_table[column].values)
        fig, axs = plt.subplots(1, 1, figsize=(15, 7), sharey=True)
        axs.bar(names, values)
        axs.set_xticks(names)
        axs.set_xlabel('Cluster ID')
        axs.set_ylabel('Probability')
        fig.suptitle(column + ' for each cluster')
        plt.show()

    def plot_feature_prob_in_clusters(self):
        for column in self.binary_columns:
            self.plot_feature_probability_across_clusters(column)

    
    

    