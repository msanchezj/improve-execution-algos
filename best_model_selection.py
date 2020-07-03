from hmmlearn.hmm import GaussianHMM, GMMHMM, MultinomialHMM
from dateutil.relativedelta import relativedelta
from itertools import chain, combinations
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from datetime import datetime

import pandas as pd
import numpy as np

class EvaluatedGMM:
    
    def __init__(self, n_components, covariance_type, max_iter, train_data, test_data, pca_flag=True):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.train_data = train_data[train_data.index.year < train_data.index[-1].year]
        self.validation_data = train_data[train_data.index.year >= train_data.index[-1].year]
        self.test_data = test_data
        self.scaler = MinMaxScaler().fit(self.train_data.values)
        self.features = train_data.columns
        if pca_flag:
            self.pca = PCA(n_components=2).fit(self.scaler.transform(self.train_data.values))
        else:
            self.pca = None
        self.gmm = GaussianMixture(n_components=self.n_components, 
                                   covariance_type=self.covariance_type, 
                                   max_iter=self.max_iter, 
                                   n_init=2)
        
    def fit_model(self):
        if self.pca is not None:
            self.X_train = self.pca.transform(self.scaler.transform(self.train_data.values))
        else:
            self.X_train = self.scaler.transform(self.train_data.values)
        self.gmm_fitted = self.gmm.fit(self.X_train)
        
    def compute_silhouette_score(self):
        if self.pca is not None:
            validation = self.pca.transform(self.scaler.transform(self.validation_data.values))
        else:
            validation = self.scaler.transform(self.validation_data.values)
        
        prediction = self.gmm_fitted.predict(validation)
        labels = np.unique(prediction)
        if len(labels) < 2:
            self.silhouette_score = 0
        else:
            try:
                self.silhouette_score = silhouette_score(validation, prediction)
            except ValueError as e:
                self.silhouette_score = 0   
        
        return self.silhouette_score
    
    def get_aic(self):
        if self.pca is not None:
            validation = self.pca.transform(self.scaler.transform(self.train_data.values))
        else:
            validation = self.scaler.transform(self.train_data.values)
        
        return self.gmm_fitted.aic(validation)
    
    def get_bic(self):
        if self.pca is not None:
            validation = self.pca.transform(self.scaler.transform(self.train_data.values))
        else:
            validation = self.scaler.transform(self.train_data.values)
            
        return self.gmm_fitted.bic(validation)
    
    def get_mean_bic_aic(self):
        
        return (self.get_bic()+self.get_aic())/2
    
    def get_params(self):
        dict_params = {'n_components': self.n_components, 
                       'covariance_type': self.covariance_type, 
                       'max_iter': self.max_iter}
        
        return dict_params
    
    def get_features(self):
        return self.features
    
class EvaluatedGaussianHMM:
    
    def __init__(self, n_components, covariance_type, max_iter, algo_type, train_data, test_data, pca_flag=True):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.algo_type = algo_type
        self.train_data = train_data[train_data.index.year < train_data.index[-1].year]
        self.validation_data = train_data[train_data.index.year >= train_data.index[-1].year]
        self.test_data = test_data
        self.scaler = MinMaxScaler().fit(self.train_data.values)
        self.features = train_data.columns
        if pca_flag:
            self.pca = PCA(n_components=2).fit(self.scaler.transform(self.train_data.values))
        else:
            self.pca = None
        self.gaussianHMM = GaussianHMM(n_components=self.n_components, 
                                       covariance_type=self.covariance_type, 
                                       n_iter=self.max_iter, 
                                       algorithm=self.algo_type)
        
    def fit_model(self):
        if self.pca is not None:
            self.X_train = self.pca.transform(self.scaler.transform(self.train_data.values))
        else:
            self.X_train = self.scaler.transform(self.train_data.values)
        self.gaussianHMM_fitted = self.gaussianHMM.fit(self.X_train)
        
    def compute_silhouette_score(self):
        if self.pca is not None:
            validation = self.pca.transform(self.scaler.transform(self.validation_data.values))
        else:
            validation = self.scaler.transform(self.validation_data.values)
        
        prediction = self.gaussianHMM_fitted.predict(validation)
        labels = np.unique(prediction)
        if len(labels) < 2:
            self.silhouette_score = 0
        else:
            try:
                self.silhouette_score = silhouette_score(validation, prediction)
            except ValueError as e:
                self.silhouette_score = 0  
        
        return self.silhouette_score
    
    def get_number_parameters(self):
        _, n_features = self.gaussianHMM_fitted.means_.shape
        if self.gaussianHMM_fitted.covariance_type == 'full':
            cov_params = self.gaussianHMM_fitted.n_components * n_features * (n_features + 1) / 2.
        elif self.gaussianHMM_fitted.covariance_type == 'diag':
            cov_params = self.gaussianHMM_fitted.n_components * n_features
        elif self.gaussianHMM_fitted.covariance_type == 'tied':
            cov_params = n_features * (n_features + 1) / 2.
        elif self.gaussianHMM_fitted.covariance_type == 'spherical':
            cov_params = self.gaussianHMM_fitted.n_components
        mean_params = n_features * self.gaussianHMM_fitted.n_components
        
        return int(cov_params + mean_params + self.gaussianHMM_fitted.n_components - 1)
    
    def get_aic(self):
        if self.pca is not None:
            validation = self.pca.transform(self.scaler.transform(self.train_data.values))
        else:
            validation = self.scaler.transform(self.train_data.values)
        aic = -2 * self.gaussianHMM_fitted.score(validation) * validation.shape[0] + 2 * self.get_number_parameters()
        
        return aic
    
    def get_bic(self):
        if self.pca is not None:
            validation = self.pca.transform(self.scaler.transform(self.train_data.values))
        else:
            validation = self.scaler.transform(self.train_data.values)
        bic = (-2 * self.gaussianHMM_fitted.score(validation) * validation.shape[0] + 
               self.get_number_parameters() * np.log(validation.shape[0]))
        
        return bic
    
    def get_mean_bic_aic(self):
        
        return (self.get_bic()+self.get_aic())/2
    
    def get_params(self):
        dict_params = {'n_components': self.n_components, 
                       'covariance_type': self.covariance_type, 
                       'max_iter': self.max_iter,
                       'algorithm': self.algo_type}
        
        return dict_params
    
    def get_features(self):
        return self.features
    
class EvaluatedBayesianGM:
    
    def __init__(self, n_components, covariance_type, max_iter, weight_concentration, train_data, test_data, pca_flag=True):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.weight_concentration = weight_concentration
        self.train_data = train_data[train_data.index.year < train_data.index[-1].year]
        self.validation_data = train_data[train_data.index.year >= train_data.index[-1].year]
        self.test_data = test_data
        self.scaler = MinMaxScaler().fit(self.train_data.values)
        self.features = train_data.columns
        if pca_flag:
            self.pca = PCA(n_components=2).fit(self.scaler.transform(self.train_data.values))
        else:
            self.pca = None
        self.bayesianGM = BayesianGaussianMixture(n_components=self.n_components, 
                                                  covariance_type=self.covariance_type, 
                                                  max_iter=self.max_iter, 
                                                  weight_concentration_prior_type=self.weight_concentration)
        
    def fit_model(self):
        if self.pca is not None:
            self.X_train = self.pca.transform(self.scaler.transform(self.train_data.values))
        else:
            self.X_train = self.scaler.transform(self.train_data.values)
        self.bayesianGM_fitted = self.bayesianGM.fit(self.X_train)
        
    def compute_silhouette_score(self):
        if self.pca is not None:
            validation = self.pca.transform(self.scaler.transform(self.validation_data.values))
        else:
            validation = self.scaler.transform(self.validation_data.values)
        
        prediction = self.bayesianGM_fitted.predict(validation)
        labels = np.unique(prediction)
        if len(labels) < 2:
            self.silhouette_score = 0
        else:
            try:
                self.silhouette_score = silhouette_score(validation, prediction)
            except ValueError as e:
                self.silhouette_score = 0
        
        return self.silhouette_score
    
    def get_number_parameters(self):
        _, n_features = self.bayesianGM_fitted.means_.shape
        if self.bayesianGM_fitted.covariance_type == 'full':
            cov_params = self.bayesianGM_fitted.n_components * n_features * (n_features + 1) / 2.
        elif self.bayesianGM_fitted.covariance_type == 'diag':
            cov_params = self.bayesianGM_fitted.n_components * n_features
        elif self.bayesianGM_fitted.covariance_type == 'tied':
            cov_params = n_features * (n_features + 1) / 2.
        elif self.bayesianGM_fitted.covariance_type == 'spherical':
            cov_params = self.bayesianGM_fitted.n_components
        mean_params = n_features * self.bayesianGM_fitted.n_components
        
        return int(cov_params + mean_params + self.bayesianGM_fitted.n_components - 1)
    
    def get_aic(self):
        if self.pca is not None:
            validation = self.pca.transform(self.scaler.transform(self.train_data.values))
        else:
            validation = self.scaler.transform(self.train_data.values)
        aic = -2 * self.bayesianGM_fitted.score(validation) * validation.shape[0] + 2 * self.get_number_parameters()
        
        return aic
    
    def get_bic(self):
        if self.pca is not None:
            validation = self.pca.transform(self.scaler.transform(self.train_data.values))
        else:
            validation = self.scaler.transform(self.train_data.values)
        bic = (-2 * self.bayesianGM_fitted.score(validation) * validation.shape[0] + 
               self.get_number_parameters() * np.log(validation.shape[0]))
        
        return bic
    
    def get_mean_bic_aic(self):
        
        return (self.get_bic()+self.get_aic())/2
    
    def get_params(self):
        dict_params = {'n_components': self.n_components, 
                       'covariance_type': self.covariance_type, 
                       'max_iter': self.max_iter, 
                       'weight_concentration_prior_type': self.weight_concentration}
        
        return dict_params
    
    def get_features(self):
        return self.features

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(3, len(s)+1))

def get_features_df(list_of_features):
    features_df = pd.concat(list_of_features, axis=1).dropna()
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    features_df = features_df.dropna()
    features_df = features_df[features_df != 0.0].shift(1).dropna()

    return features_df

def split_train_test_data(df, size_in_years):
    training_data = df[df.index[0]:df.index[-1] - relativedelta(years=size_in_years, hours=-9, minutes = -5)]
    test_data = df[df.index[-1] - relativedelta(years=size_in_years, hours=-9):]

    return training_data, test_data

def get_best_gmm_aic_bic(list_of_features, pca_flag=True):
    n_components = [2,3,4]
    cov_type = ['full', 'diag', 'spherical', 'tied']
    max_iter = [100, 200, 400, 800]

    dict_sc = {}
    dict_features = {}
    dict_bic = {}
    dict_aic = {}
    dict_score = {}

    for features_combination in list(powerset(list_of_features)):
        features_df = get_features_df(features_combination)
        train_data, test_data = split_train_test_data(features_df, 2)
        for n_comp in n_components:
            for cov in cov_type:
                for iters in max_iter:
                    model = EvaluatedGMM(n_comp, cov, iters, train_data, test_data, pca_flag)
                    model.fit_model()
                    mean_bic_aic = model.get_mean_bic_aic()
                    dict_bic[mean_bic_aic] = model.get_bic()
                    dict_aic[mean_bic_aic] = model.get_aic()
                    sc = model.compute_silhouette_score()
                    dict_sc[mean_bic_aic] = sc
                    dict_score[mean_bic_aic] = model.get_params()
                    dict_features[mean_bic_aic] = model.get_features()

    max_sc = max(list(dict_sc.values()))
    min_bic_aic = min(list(dict_score.keys()))
    aic = dict_aic.get(min_bic_aic)
    bic = dict_bic.get(min_bic_aic)
    silhouette = dict_sc.get(min_bic_aic) 
    max_params = dict_score.get(min_bic_aic)
    features_used = dict_features.get(min_bic_aic)
    print("Optimal params are {} using {} obtaining a Combined BIC-AIC score of {}".format(max_params, list(features_used), min_bic_aic))
    print("Scores obtained -> BIC: {}, AIC: {}, SC: {}".format(bic, aic, silhouette))

    best_model = GaussianMixture(n_components=max_params.get('n_components'), 
                                 covariance_type=max_params.get('covariance_type'), 
                                 max_iter=max_params.get('max_iter'), 
                                 n_init=3)
    
    final_scores_df = {'bic': bic, 'aic': aic, 'sc': sc}
    
    return best_model, features_used, final_scores_df;

def get_best_ghmm_aic_bic(list_of_features, pca_flag=True):
    n_components = [2,3,4]
    cov_type = ['full', 'diag', 'spherical', 'tied']
    max_iter = [100, 200, 400, 800]
    algorithms = ['viterbi', 'map']
    
    dict_sc = {}
    dict_features = {}
    dict_bic = {}
    dict_aic = {}
    dict_score = {}

    for features_combination in list(powerset(list_of_features)):
        features_df = get_features_df(features_combination)
        train_data, test_data = split_train_test_data(features_df, 2)
        for n_comp in n_components:
            for cov in cov_type:
                for iters in max_iter:
                    for algo in algorithms:
                        model = EvaluatedGaussianHMM(n_comp, cov, iters, algo, train_data, test_data, pca_flag)
                        model.fit_model()
                        mean_bic_aic = model.get_mean_bic_aic()
                        dict_bic[mean_bic_aic] = model.get_bic()
                        dict_aic[mean_bic_aic] = model.get_aic()
                        sc = model.compute_silhouette_score()
                        dict_sc[mean_bic_aic] = sc
                        dict_score[mean_bic_aic] = model.get_params()
                        dict_features[mean_bic_aic] = model.get_features()

    max_sc = max(list(dict_sc.values()))
    min_bic_aic = min(list(dict_score.keys()))
    aic = dict_aic.get(min_bic_aic)
    bic = dict_bic.get(min_bic_aic)
    silhouette = dict_sc.get(min_bic_aic) 
    max_params = dict_score.get(min_bic_aic)
    features_used = dict_features.get(min_bic_aic)
    print("Optimal params are {} using {} obtaining a Combined BIC-AIC score of {}".format(max_params, list(features_used), min_bic_aic))
    print("Scores obtained -> BIC: {}, AIC: {}, SC: {}".format(bic, aic, silhouette))

    best_model = GaussianHMM(n_components=max_params.get('n_components'), 
                             covariance_type=max_params.get('covariance_type'),
                             n_iter=max_params.get('max_iter'), 
                             algorithm=max_params.get('algorithm'))
    
    final_scores_df = {'bic': bic, 'aic': aic, 'sc': sc}
    
    return best_model, features_used, final_scores_df;

def get_best_bgm_aic_bic(list_of_features, pca_flag=True):
    n_components = [2,3,4]
    cov_type = ['full', 'diag', 'spherical', 'tied']
    max_iter = [100, 200, 400, 800]
    w_c_types = ['dirichlet_process', 'dirichlet_distribution']
          
    dict_sc = {}
    dict_features = {}
    dict_bic = {}
    dict_aic = {}
    dict_score = {}

    for features_combination in list(powerset(list_of_features)):
        features_df = get_features_df(features_combination)
        train_data, test_data = split_train_test_data(features_df, 2)
        for n_comp in n_components:
            for cov in cov_type:
                for iters in max_iter:
                    for w_c_type in w_c_types:
                        model = EvaluatedBayesianGM(n_comp, cov, iters, w_c_type, train_data, test_data, pca_flag)
                        model.fit_model()
                        mean_bic_aic = model.get_mean_bic_aic()
                        dict_bic[mean_bic_aic] = model.get_bic()
                        dict_aic[mean_bic_aic] = model.get_aic()
                        sc = model.compute_silhouette_score()
                        dict_sc[mean_bic_aic] = sc
                        dict_score[mean_bic_aic] = model.get_params()
                        dict_features[mean_bic_aic] = model.get_features()

    max_sc = max(list(dict_sc.values()))
    min_bic_aic = min(list(dict_score.keys()))
    aic = dict_aic.get(min_bic_aic)
    bic = dict_bic.get(min_bic_aic)
    silhouette = dict_sc.get(min_bic_aic) 
    max_params = dict_score.get(min_bic_aic)
    features_used = dict_features.get(min_bic_aic)
    print("Optimal params are {} using {} obtaining a Combined BIC-AIC score of {}".format(max_params, list(features_used), min_bic_aic))
    print("Scores obtained -> BIC: {}, AIC: {}, SC: {}".format(bic, aic, silhouette))

    best_model = BayesianGaussianMixture(n_components=max_params.get('n_components'), 
                             covariance_type=max_params.get('covariance_type'),
                             max_iter=max_params.get('max_iter'), 
                             weight_concentration_prior_type=max_params.get('weight_concentration_prior_type'))
    
    final_scores_df = {'bic': bic, 'aic': aic, 'sc': sc}
    
    return best_model, features_used, final_scores_df;

def get_best_gmm(list_of_features, pca_flag=True):
    
    n_components = [2,3,4]
    cov_type = ['full', 'diag', 'spherical', 'tied']
    max_iter = [100, 200, 400, 800]

    dict_scores = {}
    dict_features = {}
    dict_bic = {}
    dict_aic = {}

    for features_combination in list(powerset(list_of_features)):
        features_df = get_features_df(features_combination)
        train_data, test_data = split_train_test_data(features_df, 2)
        for n_comp in n_components:
            for cov in cov_type:
                for iters in max_iter:
                    model = EvaluatedGMM(n_comp, cov, iters, train_data, test_data, pca_flag)
                    model.fit_model()
                    sc = model.compute_silhouette_score()
                    dict_scores[sc] = model.get_params()
                    dict_features[sc] = model.get_features()
                    dict_aic[sc] = model.get_aic()
                    dict_bic[sc] = model.get_bic()

    max_sc = max(list(dict_scores.keys()))
    max_params = dict_scores.get(max_sc)
    features_used = dict_features.get(max_sc)
    print("Optimal params are {} using {} obtaining a Silhouette Score of {}".format(max_params, list(features_used), max_sc))

    best_model = GaussianMixture(n_components=max_params.get('n_components'), 
                                 covariance_type=max_params.get('covariance_type'), 
                                 max_iter=max_params.get('max_iter'), 
                                 n_init=100)
    
    final_scores_df = {'aic': dict_aic.get(max_sc), 'bic' : dict_aic.get(max_sc), 'sc': max_sc}
    
    return best_model, features_used, final_scores_df;

def get_best_ghmm(list_of_features, pca_flag=True):
    
    n_components = [2,3,4]
    cov_type = ['full', 'diag', 'spherical', 'tied']
    max_iter = [100, 200, 400, 800]
    algorithms = ['viterbi', 'map']
    
    dict_scores = {}
    dict_features = {}
    dict_bic = {}
    dict_aic = {}

    for features_combination in list(powerset(list_of_features)):
        features_df = get_features_df(features_combination)
        train_data, test_data = split_train_test_data(features_df, 2)
        for n_comp in n_components:
            for cov in cov_type:
                for iters in max_iter:
                    for algo in algorithms:
                        model = EvaluatedGaussianHMM(n_comp, cov, iters, algo, train_data, test_data, pca_flag)
                        model.fit_model()
                        sc = model.compute_silhouette_score()
                        dict_scores[sc] = model.get_params()
                        dict_features[sc] = model.get_features()
                        dict_aic[sc] = model.get_aic()
                        dict_bic[sc] = model.get_bic()

    max_sc = max(list(dict_scores.keys()))
    max_params = dict_scores.get(max_sc)
    features_used = dict_features.get(max_sc)
    print("Optimal params are {} using {} obtaining a Silhouette Score of {}".format(max_params, list(features_used), max_sc))

    best_model = GaussianHMM(n_components=max_params.get('n_components'), 
                             covariance_type=max_params.get('covariance_type'),
                             n_iter=max_params.get('max_iter'), 
                             algorithm=max_params.get('algorithm'))
    
    final_scores_df = {'aic': dict_aic.get(max_sc), 'bic' : dict_aic.get(max_sc), 'sc': max_sc}
    
    return best_model, features_used, final_scores_df;

def get_best_bgm(list_of_features, pca_flag=True):
    
    n_components = [2,3,4]
    cov_type = ['full', 'diag', 'spherical', 'tied']
    max_iter = [100, 200, 400, 800]
    w_c_types = ['dirichlet_process', 'dirichlet_distribution']
    
    dict_scores = {}
    dict_features = {}
    dict_bic = {}
    dict_aic = {}

    for features_combination in list(powerset(list_of_features)):
        features_df = get_features_df(features_combination)
        train_data, test_data = split_train_test_data(features_df, 2)
        for n_comp in n_components:
            for cov in cov_type:
                for iters in max_iter:
                    for w_c_type in w_c_types:
                        model = EvaluatedBayesianGM(n_comp, cov, iters, w_c_type, train_data, test_data, pca_flag)
                        model.fit_model()
                        sc = model.compute_silhouette_score()
                        dict_scores[sc] = model.get_params()
                        dict_features[sc] = model.get_features()
                        dict_aic[sc] = model.get_aic()
                        dict_bic[sc] = model.get_bic()

    max_sc = max(list(dict_scores.keys()))
    max_params = dict_scores.get(max_sc)
    features_used = dict_features.get(max_sc)
    print("Optimal params are {} using {} obtaining a Silhouette Score of {}".format(max_params, list(features_used), max_sc))

    best_model = BayesianGaussianMixture(n_components=max_params.get('n_components'), 
                             covariance_type=max_params.get('covariance_type'),
                             max_iter=max_params.get('max_iter'), 
                             weight_concentration_prior_type=max_params.get('weight_concentration_prior_type'), 
                             n_init=100)
    
    final_scores_df = {'aic': dict_aic.get(max_sc), 'bic' : dict_aic.get(max_sc), 'sc': max_sc}
    
    return best_model, features_used, final_scores_df;