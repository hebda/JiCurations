#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:44:57 2015

@author: Eric

Create predictions for future school performance
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcess
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.preprocessing import Imputer
from statsmodels.tsa.arima_model import ARIMA

import config
reload(config)
import join_data
reload(join_data)
import utilities
reload(utilities)



def main(**kwargs):
    """ Read in all data and run the fits/predictions over all school statistics separately """


    ## Read in data
    con = utilities.connect_to_sql('joined')
    with con:
        cur = con.cursor()
        data_a_d = {}
        all_Database_l = join_data.Database_l + join_data.DistrictDatabase_l
        for Database in all_Database_l:
            Instance = Database()
            feature_s = Instance.new_table_s
            field_s_l = ['ENTITY_CD'] + \
                ['{0}_{1:d}'.format(feature_s, year) for year in config.year_l]
            data_a_d[feature_s] = utilities.select_data(con, cur, field_s_l,
                                                        'master',
                                                        output_type='np_array')

    ## Run prediction over all features
    for feature_s in data_a_d.iterkeys():
        predict_a_feature(data_a_d, feature_s, **kwargs)



class AutoRegression(object):
    """ The main ARIMA object used for fitting and prediction """

    def __init__(self):
        pass

    def fit(self, raw_array, aux_data_a_d=None, diff=False, feature_s_l=[], holdout_col=0, lag=1, positive_control=False, regression_algorithm_s = 'elastic_net', **kwargs):
        """ Performs an auto-regression of a given lag on the input array. Axis 0 indexes observations (schools) and axis 1 indexes years. For holdout_col>0, the last holdout_col years of data will be withheld from the fitting, which is ideal for training the algorithm. """

        # Apply optional parameters
        if holdout_col > 0:
            raw_array = raw_array[:, :-holdout_col]
        if diff:
            array = np.diff(raw_array, 1, axis=1)
        else:
            array = raw_array

        # Create model and fit parameters
        Y = array[:, lag:].reshape(-1)
        X = np.ndarray((Y.shape[0], 0))
        for i in range(lag):
            X = np.concatenate((X, array[:, i:-lag+i].reshape(-1, 1)), axis=1)
            # Y = X_t = A_1 * X_(t-lag) + A_2 * X_(t-lag+1)) + ... + A_lag * X_(t-1) + A_(lag+1)
        if positive_control:
            X = np.concatenate((X, array[:, lag:].reshape(-1, 1)), axis=1)
        if aux_data_a_d:
            for feature_s in feature_s_l:
                if holdout_col > 0:
                    raw_array = aux_data_a_d[feature_s][:, :-holdout_col]
                else:
                    raw_array = aux_data_a_d[feature_s]
                if diff:
                    array = np.diff(raw_array, 1, axis=1)
                else:
                    array = raw_array
                for i in range(lag):
                    X = np.concatenate((X, array[:, i:-lag+i].reshape(-1, 1)), axis=1)
        estimatorX = Imputer(axis=0)
        X = estimatorX.fit_transform(X)
        estimatorY = Imputer(axis=0)
        Y = estimatorY.fit_transform(Y.reshape(-1, 1)).reshape(-1)

        if regression_algorithm_s == 'elastic_net':
            l1_ratio_l = [.1, .5, .7, .9, .95, .99, 1]
            alpha_l = np.logspace(-15, 5, num=11).tolist()
            max_iter = 1e5
            # It's too slow when I make it high, so I'll keep it low for now
            model = ElasticNetCV(l1_ratio=l1_ratio_l, alphas=alpha_l, max_iter=max_iter,
                                 fit_intercept=True, normalize=True)
        elif regression_algorithm_s == 'gaussian_process':
            model = GaussianProcess()
            # This currently gives the following error: "Exception: Multiple input features cannot have the same target value."
        elif regression_algorithm_s == 'gradient_boosting':
            model = GradientBoostingRegressor(max_features='sqrt')
        elif regression_algorithm_s == 'linear_regression':
            model = LinearRegression(fit_intercept=True, normalize=True)
        elif regression_algorithm_s == 'random_forest':
            model = RandomForestRegressor(max_features='auto')
        model.fit(X, Y)
        if regression_algorithm_s in ['elastic_net', 'linear_regression']:
            with open(os.path.join(config.plot_path, 'coeff_list.txt'), 'a') as f:
                f.write('Lag of {0:d}:\n'.format(lag))
#                f.write('\nElastic net: R^2 = %0.5f, l1_ratio = %0.2f, alpha = %0.1g' %
#                      (model.score(X, Y), model.l1_ratio_, model.alpha_))
                coeff_t = model.coef_
                assert(not positive_control) # The coefficients won't currently line up
                for i_lag in range(lag):
                    f.write('\ti_lag = {0:d}: {1:0.2g}\n'.format(lag-i_lag, coeff_t[i_lag]))
                for i_feature, feature_s in enumerate(feature_s_l):
                    for i_lag in range(lag):
                        f.write('\t{0}:\n\t\ti_lag = {1:d}: {2:0.2g}\n'.format(feature_s, lag-i_lag, coeff_t[lag*(i_feature+1) + i_lag]))

        return model

    def predict(self, raw_array, results, aux_data_a_d=None, diff=False,
                holdout_col=0, lag=1, positive_control=False, **kwargs):
        """ Given the input results model, predicts the year of data immediately succeeding the last year of the input array. Axis 0 indexes observations (schools) and axis 1 indexes years. For holdout_col>0, the last holdout_col years of data will be withheld from the prediction, which is ideal for finding the error of the algorithm. """

        if positive_control:
            if holdout_col > 0:
                if diff:
                    if holdout_col == 1:
                        control_array = np.diff(raw_array[:, -2:],
                                    1, axis=1)
                    else:
                        control_array = \
                            np.diff(raw_array[:, -holdout_col-1:-holdout_col+1],
                                    1, axis=1)
                else:
                    control_array = raw_array[:, -holdout_col]
            else:
                control_array = np.random.randn(raw_array.shape[0], 1)

        if holdout_col > 0:
            raw_array = raw_array[:, :-holdout_col]
        prediction_raw_array = raw_array
        if diff:
            array = np.diff(raw_array, 1, axis=1)
            X = array[:, -lag:]
            if positive_control:
                X = np.concatenate((X, control_array.reshape(-1, 1)), axis=1)
            if aux_data_a_d:
                for feature_s in aux_data_a_d.iterkeys():
                    if holdout_col > 0:
                        raw_array = aux_data_a_d[feature_s][:, :-holdout_col]
                    else:
                        raw_array = aux_data_a_d[feature_s]
                    array = np.diff(raw_array, 1, axis=1)
                    X = np.concatenate((X, array[:, -lag:]), axis=1)
            estimatorX = Imputer(axis=0)
            X = estimatorX.fit_transform(X)
            predicted_change_a = results.predict(X)
            estimator_orig = Imputer(axis=0)
            orig_a = estimator_orig.fit_transform(prediction_raw_array[:, -1].reshape(-1,1))
            prediction_a = orig_a + predicted_change_a.reshape(-1, 1)
        else:
            array = raw_array
            X = array[:, -lag:]
            if positive_control:
                X = np.concatenate((X, control_array.reshape(-1, 1)), axis=1)
            if aux_data_a_d:
                for feature_s in aux_data_a_d.iterkeys():
                    if holdout_col > 0:
                        raw_array = aux_data_a_d[feature_s][:, :-holdout_col]
                    else:
                        raw_array = aux_data_a_d[feature_s]
                    array = raw_array
                    X = np.concatenate((X, array[:, -lag:]), axis=1)
            estimatorX = Imputer(axis=0)
            X = estimatorX.fit_transform(X)
            prediction_a = results.predict(X)

        return prediction_a.reshape((-1, 1))



class IndependentAutoRegression(object):
    """ What happens if you assume that each school has its own behavior over time and fit an auto-regressive model to each school separately? The answer is that you get very noisy predictions. """


    def __init__(self):
        pass


    def fit(self, raw_array, holdout_col=0, lag=1, **kwargs):
        """ Performs an auto-regression of a given lag on the input array, to each school separately. Axis 0 indexes observations (schools) and axis 1 indexes years. For holdout_col>0, the last holdout_col years of data will be withheld from the fitting, which is ideal for training the algorithm. """

        # Apply optional parameters
        if holdout_col > 0:
            raw_array = raw_array[:, :-holdout_col]
        array = raw_array

        # Create model and fit parameters
        estimator = Imputer(axis=0)
        array = estimator.fit_transform(array)
        model_l = []
        for school_a in array:
            Y = school_a.reshape(-1)[lag:].reshape(-1)
            X = np.ndarray((Y.shape[0], 0))
            for i in range(lag):
                X = np.concatenate((X, school_a[i:-lag+i].reshape(-1, 1)), axis=1)
            model = LinearRegression(fit_intercept=True)
            model.fit(X,Y)
            model_l.append(model)

        return model_l


    def predict(self, array, results_l,
                holdout_col=0, lag=1, **kwargs):
        """ Given the input results model, predicts the year of data immediately succeeding the last year of the input array. Axis 0 indexes observations (schools) and axis 1 indexes years. For holdout_col>0, the last holdout_col years of data will be withheld from the prediction, which is ideal for finding the error of the algorithm. """

        if holdout_col > 0:
            array = array[:, :-holdout_col]

        X = array[:, -lag:]
        estimatorX = Imputer(axis=0)
        X = estimatorX.fit_transform(X)
        prediction_l = []
        for i_school, school_a in enumerate(X):
            prediction = results_l[i_school].predict(school_a)
            prediction_l.append(prediction)

        return np.array(prediction_l).reshape((-1, 1))



class MeanOverYears(object):
    """ Baseline: assume that next year's value will be the same as the mean over the previous years, a.k.a. the historical average """

    def __init__(self):
        pass

    def fit(self, array, **kwargs):
        return None

    def predict(self, array, results, holdout_col=0, **kwargs):
        if holdout_col > 0:
            array = array[:, :-holdout_col]
        mean_over_years_a = np.mean(array, axis=1)
        return mean_over_years_a.reshape((-1, 1))



class SameAsLastYear(object):
    """ Baseline: assume that next year's value will be the same as last year's value """

    def __init__(self):
        pass

    def fit(self, array, **kwargs):
        return None

    def predict(self, array, results, holdout_col=0, **kwargs):
        if holdout_col > 0:
            array = array[:, :-holdout_col]
        last_year_a = array[:, -1]
        return last_year_a.reshape((-1, 1))



class SameChangeAsLastYear(object):
    """ Baseline: assume that the change between this year and next year's values will be the same as the change between last year and this year's values """

    def __init__(self):
        pass

    def fit(self, array, **kwargs):
        return None

    def predict(self, array, results, holdout_col=0, **kwargs):
        if holdout_col > 0:
            array = array[:, :-holdout_col]
        change_over_last_year_a = array[:, -1] - array[:, -2]
        extrapolation_from_last_year_a = array[:, -1] + change_over_last_year_a
        return extrapolation_from_last_year_a.reshape((-1, 1))



def find_rms_error(Y, prediction_Y,
                   plot_residuals=False, residual_plot_name_s='',
                   **kwargs):
    """ Calculate the root-mean-squared error between the actual data and the prediction """

    assert(Y.shape == prediction_Y.shape)
    valid_col_a = ~np.isnan(Y)
    residual_a = Y[valid_col_a] - prediction_Y[valid_col_a]
    rmse = np.sqrt(np.mean((residual_a)**2))

    if plot_residuals:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        try:
            ax.hist(residual_a, bins=20)
        except:
            pass
        ax.set_xlabel('Residual')
        ax.set_ylabel('Frequency')
        save_path = os.path.join(config.plot_path, 'residual_histogram')
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        plt.savefig(os.path.join(save_path, residual_plot_name_s + '.png'))

    return rmse



def fit_and_predict(array, Class, feature_s, model_s, aux_data_a_d=None, **kwargs):
    """ Given an array, fits it using the regression technique specified by Class and the optional options. """

    num_years_to_predict = len(config.prediction_year_l)

    instance = Class()
    results_d = {}


    ## Training set: all years before the most recent

    # Fit on all years before the most recent
    with open(os.path.join(config.plot_path, 'coeff_list.txt'), 'a') as f:
        f.write('(fit on all years but last)\n')
    results_d['result_object'] = instance.fit(array, holdout_col=1,
                                              aux_data_a_d=aux_data_a_d,
                                              **kwargs)

    # Measure training set error
    last_fitted_year_prediction_a = \
        instance.predict(array, results_d['result_object'],
                         aux_data_a_d=aux_data_a_d,
                         holdout_col=2,
                         **kwargs)
    result_s = 'train_rms_error'
    residual_plot_name_s = feature_s + '__' + model_s + '__' + result_s
    results_d[result_s] = find_rms_error(array[:, -2].reshape(-1),
        last_fitted_year_prediction_a.reshape(-1),
        residual_plot_name_s=residual_plot_name_s,
        **kwargs)

    # Measure error of most recent year of data as a test set
    last_data_year_prediction_a = \
        instance.predict(array, results_d['result_object'],
                         aux_data_a_d=aux_data_a_d,
                         holdout_col=1,
                         **kwargs)
    result_s = 'test_rms_error'
    residual_plot_name_s = feature_s + '__' + model_s + '__' + result_s
    results_d[result_s] = find_rms_error(array[:, -1].reshape(-1),
        last_data_year_prediction_a.reshape(-1),
        residual_plot_name_s=residual_plot_name_s,
        **kwargs)

    # Predict future values
    future_prediction_a = np.ndarray((array.shape[0], 0))
    for year in range(num_years_to_predict):
        combined_data_a = np.concatenate((array, future_prediction_a), axis=1)
        new_year_prediction_a = instance.predict(combined_data_a,
                                                 results_d['result_object'],
                                                 aux_data_a_d=aux_data_a_d,
                                                 **kwargs)
        future_prediction_a = np.concatenate((future_prediction_a,
                                              new_year_prediction_a), axis=1)
    results_d['prediction_a'] = future_prediction_a


    ## Perform 10-fold cross validation

    kf = cross_validation.KFold(array.shape[0], n_folds=10, shuffle=True)
    cross_val_train_rmse_l = []
    cross_val_test_rmse_l = []
    for train_index, test_index in kf:
#        print('TRAIN:', len(train_index), 'TEST:', len(test_index))

        # Creating new aux dicts
        if aux_data_a_d:
            aux_train_a_d = {}
            aux_test_a_d = {}
            for key, val in aux_data_a_d.iteritems():
                aux_train_a_d[key] = val[train_index, :]
                aux_test_a_d[key] = val[test_index, :]
        else:
            aux_train_a_d = None
            aux_test_a_d = None


        # Train model
        instance = Class()
        with open(os.path.join(config.plot_path, 'coeff_list.txt'), 'a') as f:
            f.write('(10-fold CV fits)\n')
        result = instance.fit(array[train_index, :],
                              holdout_col=0,
                              aux_data_a_d=aux_train_a_d,
                              **kwargs)

        # Find train RMSE
        train_prediction_a = instance.predict(array[train_index, :],
                                        result,
                                        aux_data_a_d=aux_train_a_d,
                                        holdout_col=1,
                                        **kwargs)
        cross_val_train_rmse_l.append(find_rms_error(array[train_index, -1].reshape(-1),
            train_prediction_a.reshape(-1), **kwargs))

        # Find test RMSE
        test_prediction_a = instance.predict(array[test_index, :],
                                        result,
                                        aux_data_a_d=aux_test_a_d,
                                        holdout_col=1,
                                        **kwargs)
        cross_val_test_rmse_l.append(find_rms_error(array[test_index, -1].reshape(-1),
            test_prediction_a.reshape(-1), **kwargs))

#    print('Cross-val train RMSE:', cross_val_train_rmse_l)
#    print('Cross-val test RMSE:', cross_val_test_rmse_l)
    results_d['cross_val_train_rms_error'] = np.mean(cross_val_train_rmse_l)
    results_d['cross_val_test_rms_error'] = np.mean(cross_val_test_rmse_l)


    ## Validating based on RMSE of prediction 3 years out

    # Fit on all years before the most recent 3
    with open(os.path.join(config.plot_path, 'coeff_list.txt'), 'a') as f:
        f.write('(fit on all years but last 3)\n')
    instance = Class()
    result = instance.fit(array, holdout_col=3,
                          aux_data_a_d=aux_data_a_d,
                          **kwargs)

    # Measure training set error
    three_year_train_prediction_a = \
        instance.predict(array, result,
                         aux_data_a_d=aux_data_a_d,
                         holdout_col=4,
                         **kwargs)
    result_s = 'three_year_train_rms_error'
    residual_plot_name_s = feature_s + '__' + model_s + '__' + result_s
    results_d[result_s] = find_rms_error(array[:, -4].reshape(-1),
        three_year_train_prediction_a.reshape(-1),
        residual_plot_name_s=residual_plot_name_s,
        **kwargs)

    # Measure error of most recent year of data as a test set
    future_prediction_a = np.ndarray((array.shape[0], 0))
    for year in range(3):
        combined_data_a = np.concatenate((array[:, :-3], future_prediction_a), axis=1)
        new_year_prediction_a = instance.predict(combined_data_a,
                                                 result,
                                                 aux_data_a_d=aux_data_a_d,
                                                 **kwargs)
        future_prediction_a = np.concatenate((future_prediction_a,
                                              new_year_prediction_a), axis=1)
    result_s = 'three_year_test_rms_error'
    residual_plot_name_s = feature_s + '__' + model_s + '__' + result_s
    results_d[result_s] = find_rms_error(array[:, -1].reshape(-1),
        future_prediction_a[:, -1].reshape(-1),
        residual_plot_name_s=residual_plot_name_s,
        **kwargs)

    return results_d



def predict_a_feature(input_data_a_d, primary_feature_s,
                      aux_features=True, save_data=False,
                      **kwargs):
    """ Wraps around fit_and_predict: runs various regression models on the input school statistic and outputs/plots the results """

    print('\n\nStarting prediction for {0}.\n'.format(primary_feature_s))
    with open(os.path.join(config.plot_path, 'coeff_list.txt'), 'a') as f:
        f.write('\n\nStarting prediction for {0}.\n'.format(primary_feature_s))

    data_a_d = input_data_a_d.copy()
    index_a = data_a_d[primary_feature_s][:, 0]


    ## Drop the ENTITY_CD column
    for feature_s in data_a_d.iterkeys():
        data_a_d[feature_s] = data_a_d[feature_s][:, 1:]


    ## Split data
    main_data_a = data_a_d[primary_feature_s]
    data_a_d.pop(primary_feature_s)
    feature_s_l = sorted(data_a_d.keys())
    if not aux_features:
        data_a_d = {}
        feature_s_l = []


    ## Run regression models, validate and predict future scores, and run controls
    all_results_d = {}

    # Run autoregression with different lags on raw test scores
    lag_l = range(1, 5)
    for lag in lag_l:
        model_s = 'raw_lag{:d}'.format(lag)
        print(model_s + ':')
        all_results_d[model_s] = fit_and_predict(main_data_a, AutoRegression,
                                                 primary_feature_s,
                                                 model_s,
                                                 aux_data_a_d=data_a_d,
                                                 diff=False,
                                                 feature_s_l=feature_s_l,
                                                 lag=lag,
                                                 **kwargs)

#    # Run autogression with different lags on diff of test scores w.r.t. year
#    lag_l = range(1, 4)
#    for lag in lag_l:
#        model_s = 'diff_lag{:d}'.format(lag)
#        print(model_s + ':')
#        all_results_d[model_s] = fit_and_predict(main_data_a, AutoRegression,
#                                                 primary_feature_s,
#                                                 model_s,
#                                                 aux_data_a_d=data_a_d,
#                                                 diff=True,
#                                                 feature_s_l=feature_s_l,
#                                                 lag=lag,
#                                                 **kwargs)

#    lag_l = range(1, 5)
#    for lag in lag_l:
#        model_s = 'ind_raw_lag{:d}'.format(lag)
#        print(model_s + ':')
#        all_results_d[model_s] = fit_and_predict(main_data_a, IndependentAutoRegression,
#                                                 primary_feature_s,
#                                                 model_s,
#                                                 lag=lag,
#                                                 **kwargs)

    # Run control: prediction is same as mean over years in training set
    model_s = 'z_mean_over_years_score_control'
    print(model_s + ':')
    all_results_d[model_s] = fit_and_predict(main_data_a, MeanOverYears,
                                             primary_feature_s, model_s,
                                             **kwargs)

    # Run control: prediction is same as previous year's data
    model_s = 'z_same_as_last_year_score_control'
    print(model_s + ':')
    all_results_d[model_s] = fit_and_predict(main_data_a, SameAsLastYear,
                                             primary_feature_s, model_s,
                                             **kwargs)

    # Run control: prediction is same as previous year's data
    model_s = 'z_same_change_as_last_year_score_control'
    print(model_s + ':')
    all_results_d[model_s] = fit_and_predict(main_data_a, SameChangeAsLastYear,
                                             primary_feature_s, model_s,
                                             **kwargs)

    chosen_baseline_s_l = ['z_mean_over_years_score_control',
                           'z_same_as_last_year_score_control']
    all_train_mses_d = {key: value['cross_val_train_rms_error'] for (key, value) in all_results_d.iteritems()}
    all_test_mses_d = {key: value['cross_val_test_rms_error'] for (key, value) in all_results_d.iteritems()}
    all_models_s_l = sorted(all_results_d.keys())
    with open(os.path.join(config.plot_path, 'RMSE_list.txt'), 'a') as f:
        f.write('\n\n{0}:\n'.format(primary_feature_s))
        for model_s in all_models_s_l:
            f.write('\n{0}:\n'.format(model_s))
            f.write('\tTrain RMSE:\n\t\t{:1.5f}\n'.format(all_train_mses_d[model_s]))
            f.write('\tTest RMSE:\n\t\t{:1.5f}\n'.format(all_test_mses_d[model_s]))
            for chosen_baseline_s in chosen_baseline_s_l:
                f.write('\t{0}: \n\t\t{1:1.5g}\n'.format(chosen_baseline_s,
                      all_test_mses_d[model_s]/all_test_mses_d[chosen_baseline_s]))


    ## Plot MSEs of all regression models

    model_s_l = sorted(all_train_mses_d.keys())
    fig = plt.figure(figsize=(1.5*len(model_s_l),12))
    ax = fig.add_axes([0.10, 0.40, 0.80, 0.50])

    # Generating bar values
    value_s_l = ['train_rms_error', 'test_rms_error',
               'cross_val_train_rms_error', 'cross_val_test_rms_error',
               'three_year_train_rms_error', 'three_year_test_rms_error']
    value_l_d = {}
    for value_s in value_s_l:
        value_l_d[value_s] = [all_results_d[iter_model_s][value_s]\
                              for iter_model_s in model_s_l]

    # Generate bar positions
    bar_width = 0.12
    value_position_l_d = {}
    for i_value, value_s in enumerate(value_s_l):
        value_position_l_d[value_s] = np.arange(len(model_s_l)) + (i_value-3)*bar_width

    # Generate colors
    value_color_l = ['r', 'y', 'g', 'c', 'b', 'm']

    # Plot bars
    for i_value, value_s in enumerate(value_s_l):
        ax.bar(value_position_l_d[value_s], value_l_d[value_s], bar_width,
               color=value_color_l[i_value], label=value_s)

    # Formatting
    ax.set_title('Comparison of RMS error of autoregression algorithms vs. controls')
    ax.set_xticks(np.arange(len(model_s_l)))
    ax.set_xticklabels(model_s_l, rotation=90)
    ax.set_ylabel('Root mean squared error')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.50))
    for chosen_baseline_s in chosen_baseline_s_l:
        ax.axhline(y=all_test_mses_d[chosen_baseline_s], color=(0.5, 0.5, 0.5))
    ax.set_ylim([0, 1.5*all_test_mses_d['z_mean_over_years_score_control']])
    save_path = os.path.join(config.plot_path, 'create_predictions')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    plt.savefig(os.path.join(save_path,
                             'rms_error_all_models__{0}.png'.format(primary_feature_s)))


    ## Save data to the SQL database
    if save_data:
        model_to_save_s = 'raw_lag1'
        new_column_s_l = ['ENTITY_CD'] + \
            ['{0}_prediction_{1:d}'.format(primary_feature_s, year)
             for year in config.prediction_year_l]
        prediction_a = np.concatenate((index_a.reshape(-1, 1),
                                       all_results_d[model_to_save_s]['prediction_a']),
                                      axis=1)
        prediction_df = pd.DataFrame(prediction_a, columns=new_column_s_l)
        utilities.write_to_sql_table(prediction_df,
                                     '{0}_prediction'.format(primary_feature_s), 'joined')



if __name__ == '__main__':
    main(aux_features=False,
         positive_control=False,
         regression_algorithm_s='linear_regression',
         plot_residuals=False,
         save_data=False)
