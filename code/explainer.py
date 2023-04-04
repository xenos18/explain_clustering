import numpy as np
import pandas as pd
from scipy.stats import beta
import matplotlib.pyplot as plt


#TO DO: модуль для определение категориальных признаков


class BayesExplainer():

    def __init__(self, data, label_col, cat_features, num_features, threshold=0.7, num_samples=20000):
        #определение кластерной матрицы
        self.data = data
        self.label_col = label_col
        self.num_clusters = data['label_col'].nunique()
        self.cat_features = cat_features
        self.num_features = num_features
        self.threshhold = threshold
        self.num_samples = num_samples
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

        # num_clusters = 10
        # num_samples = 20000
        # threshhold = 0.7

        #создаем датафрейм для сохранения промежуточных рассчетов
        df_bayesian_analysis = pd.DataFrame(index=self.cluster_table.index)
        df_bayesian_analysis['num_observations'] = self.cluster_table['num_observations']
        #создаем датафрейм для сохранения оценок значимости признака
        df_importance_scores = pd.DataFrame(index=self.cluster_table.index)

        #запускаем алгоритм для каждого признака
        for feature in self.cat_features:
            print(feature)
            #рассчитываем априорную вероятность наличия признака во всех кластерах
            df_bayesian_analysis[feature + ' conversions'] = self.cluster_table[feature]
            
            #рассчитываем апостериорную вероятность признака во всех кластерах
            df_bayesian_analysis[feature + ' posterior'] = df_bayesian_analysis[['num_observations',
                                                                        feature + ' conversions']]\
                                                                            .apply(lambda a: self.calc_posterior(a[0], a[1]),axis=1)
            #сэсплируем точки из апостериорного распределения
            df_bayesian_analysis[feature + ' samples'] = df_bayesian_analysis[feature + ' posterior']\
                .apply(lambda a: self.get_samples_from_dist(a, self.num_samples))
            
            #визуализируем апостериорное распределение на графике
            self.plot_posteriors(df_bayesian_analysis, feature)
            
            #вычисляем матрицу попарных сравнений вероятности наличия признака во всех кластерах
            posterior_comparison = self.calc_posterior_comparison_matrix(df_bayesian_analysis, feature)
            
            #оцениваем значимость признака для каждого кластера
            df_importance_scores[feature] = None
            for index in df_importance_scores.index:
                df_importance_scores[feature].loc[index] = (posterior_comparison.loc[index] >= self.threshhold).sum()
            
            #визуализируем матрицу попарных сравнений
            self.plot_posterior_matrix(posterior_comparison)
    


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
        for feature in self.cat_features:
            self.plot_feature_probability_across_clusters(feature)

    def calc_posterior(self, cl_num_observations, feature_prob, alpha_prior=1, beta_prior=1):
        #TO DO: Numerical Case
        #assuming prior - beta(1, 1) - то же самое, что и uniform on [0,1]
        """
        Calculation of posterior distribution for binary features
        """
        posterior = beta(alpha_prior+feature_prob, beta_prior+cl_num_observations-feature_prob)
        return posterior

    def get_samples_from_dist(self, posterior):
        return posterior.rvs(self.num_samples)
        
    def plot_posteriors(self, df, column, ylim=100, xmin=0, xmax=1, title=None):
        if not title:
            title = f'Posterior PDF of {column.title()}'
        plt.figure(figsize = (15, 7))
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 300
        x = np.linspace(0,1, 500)
        for i in range(1, self.num_clusters+1):
            if i > 9: #Need adaptive plotting
                plt.plot(x, df[column+ ' posterior'].loc[i].pdf(x), '.', label=f'cluster {i}')
                
            else:
                plt.plot(x, df[column+ ' posterior'].loc[i].pdf(x), label=f'cluster {i}')

        plt.ylim(0, ylim)
        plt.xlim(xmin, xmax)
        plt.xlabel('Value', fontsize=20)
        plt.ylabel('Density', fontsize=20)
        plt.title(title, fontsize=25)
        plt.rc('xtick', labelsize=20) 
        plt.rc('ytick', labelsize=20) 
        plt.legend(fontsize=20)
        plt.show()
        
    def calc_posterior_comparison_matrix(self, df, column_name):
        results_comparison = pd.DataFrame(index=list(range(1, self.num_clusters+1)), columns = list(range(1, self.num_clusters+1)))
        for row in results_comparison.index:
            for column in results_comparison.columns:
                results_comparison.loc[row, column] = (df[column_name + ' samples'].loc[row] \
                                                    > df[column_name + ' samples'].loc[column]).mean()
        results_comparison = results_comparison.astype(np.float)
        return results_comparison    

    def plot_posterior_matrix(self, df):
        plt.pcolor(df, cmap='GnBu')
        plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
        plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
        for y in range(df.shape[0]):
            for x in range(df.shape[1]):
                plt.text(x + 0.5, y + 0.5, '%.4f' % df.iloc[y, x],
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontdict={'size':5}
                        )
        plt.show()

    
    

    