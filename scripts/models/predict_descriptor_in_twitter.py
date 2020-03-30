"""
Predict descriptor for valid LOC NEs
in Twitter data using logistic regression.

Full model to test:
descriptor ~ event_importance + global_importance + time_importance + err

Results to report: 

1. joint prediction 
2. per-data prediction

and 

1. effect size, significance
2. prediction accuracy, model deviance
"""
from argparse import ArgumentParser
from datetime import datetime, timedelta
import dateutil
import pandas as pd
import pickle
import sys
if('..' not in sys.path):
    sys.path.append('..')
from data_processing.data_helpers import extract_freq_NEs
import numpy as np
from statsmodels.api import GLM
from statsmodels.formula.api import glm, logit
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import Logit
from sklearn.model_selection import KFold
import re
import os
import logging

def assign_time_period(data, event_time_vars):
    """
    Assign time period to a given data point.
    
    """
    event_time_vars_since = ['since_%s'%(x) for x in event_time_vars]
    time_period_idx = np.where(data.loc[event_time_vars_since] > 0.)[0]
    if(len(time_period_idx) > 0):
        time_period = event_time_vars[time_period_idx[-1]]
    else:
        time_period = event_time_vars[-1]
#     time_period = event_time_vars[0]
#     for etv in event_time_vars:
#         etv_since = 'since_%s'%(etv)
#         if(data.loc[etv_since] == 0.):
#             break
#         time_period = etv
    return time_period

def compute_days(data):
    data_time = data.assign(**{'date_day' : data.loc[:, 'date'].apply(lambda x: datetime(day=x.day, month=x.month, year=x.year))})
    return data_time

def clean_NE(x):
    x = x.replace('-', '_')
    return x

def load_clean_data(combined_data_file, event_date_file='../../data/hurricane_data/hurricane_dates.tsv', geo_dict_file='../../data/mined_tweets/combined_data_NE_tweets_geo_dict.pickle', dep_var='has_descriptor', N_importance_bins=4, NE_col='NE_fixed', data_group_name='data_name_fixed', top_k_NE_min=5):
    """
    Load and clean prediction data.
    """
    combined_tag_data = pd.read_csv(combined_data_file, sep='\t', compression='gzip', index_col=False, converters={'date':lambda x: dateutil.parser.parse(x)})
    combined_tag_data_valid_loc = combined_tag_data[combined_tag_data.loc[:, 'valid_loc']]
    ## round date to days
    combined_tag_data_valid_loc = compute_days(combined_tag_data_valid_loc)
    # compute timestamp because it's easier
    combined_tag_data_valid_loc = combined_tag_data_valid_loc.assign(**{'date' : combined_tag_data_valid_loc.loc[:, 'date'].apply(lambda x: x.timestamp())})
    
    ## add relevant dates
    # relevant hurricane dates
    event_dates = pd.read_csv(event_date_file, sep='\t', index_col=False, converters={'hurricane_date':lambda x: datetime.strptime('%s EST'%(x), '%d-%m-%y %Z')})
    event_dates.rename(columns={'hurricane_name':data_group_name}, inplace=True)
    event_dates_pivot = event_dates.pivot(index=data_group_name, columns='hurricane_type', values='hurricane_date').reset_index()
    combined_tag_data_valid_loc_time = pd.merge(combined_tag_data_valid_loc, event_dates_pivot, on=data_group_name)
    # remove all tweets sent before formation time
    time_var = 'date'
    combined_tag_data_valid_loc_time = combined_tag_data_valid_loc_time[combined_tag_data_valid_loc_time.loc[:, 'form'].apply(lambda x: x.timestamp()) <= combined_tag_data_valid_loc_time.loc[:, time_var]]
    combined_tag_data_valid_loc_time_ = []
    # add since_event date
    event_time_vars = ['form', 'landfall', 'dissipation']
    for data_name_i, data_i in combined_tag_data_valid_loc_time.groupby(data_group_name):
        date_min_i = data_i.loc[:, time_var].min()
        for j, x in enumerate(event_time_vars):
            x_since = 'since_%s'%(x)
            data_i.loc[:, x_since] = (data_i.loc[:, time_var] - data_i.loc[:, x].apply(lambda x: x.timestamp())).apply(lambda x: max(0, x))
            # convert to days for better interpretability
            data_i.loc[:, x_since] = data_i.loc[:, x_since] / (3600*24)
        # also get time period variable for completeness
        # ex. formation_period => tweet posted DURING formation period
        data_i = data_i.assign(**{'time_period_type' : data_i.apply(lambda x: assign_time_period(x, event_time_vars), axis=1)})
        # don't do this! creates perfect separation in data
    #     for j, x in enumerate(event_time_vars):
    #         x_since = 'since_%s'%(x)
    #         event_time_vars_after_x = ['since_%s'%(v) for v in event_time_vars[j+1:]]
    # #         logging.debug('event time vars after %s = %s'%(x_since, ','.join(event_time_vars_after_x)))
    #         if(len(event_time_vars_after_x) > 0):
    #             data_i.loc[:, '%s_binary'%(x_since)] = ((data_i.loc[:, x_since] > 0) & (data_i.loc[:, event_time_vars_after_x].sum(axis=1) == 0.)).astype(int)
    #         else:
    #             data_i.loc[:, '%s_binary'%(x_since)] = (data_i.loc[:, x_since] > 0).astype(int)
        combined_tag_data_valid_loc_time_.append(data_i)
    combined_tag_data_valid_loc_time = pd.concat(combined_tag_data_valid_loc_time_, axis=0)
    # remove invalid tweets
    # combined_tag_data_valid_loc_time = combined_tag_data_valid_loc_time[combined_tag_data_valid_loc_time.loc[:, ['since_form_binary', 'since_landfall_binary', 'since_dissipation_binary']].sum(axis=1).astype(int)==1]

    ## add importance stats
    ## log-transform and bin to handle skew and nonlinear effects
    geo_dict = pickle.load(open(geo_dict_file, 'rb'))
    max_pops = pd.Series({k : v.loc[:, 'population'].max() for k,v in geo_dict.items()})
    max_names = pd.Series({k : v.loc[:, 'alternate_name_count'].max() for k,v in geo_dict.items()})
    combined_tag_data_valid_loc_time = combined_tag_data_valid_loc_time.assign(max_pop=combined_tag_data_valid_loc_time.loc[:, NE_col].apply(lambda x : max_pops.loc[x.replace('_', ' ')]))
    combined_tag_data_valid_loc_time = combined_tag_data_valid_loc_time.assign(max_names=combined_tag_data_valid_loc_time.loc[:, NE_col].apply(lambda x : max_names.loc[x.replace('_', ' ')]))
    importance_stats = ['max_pop', 'max_names']
    combined_tag_data_importance = []
    for data_name_i, data_i in combined_tag_data_valid_loc_time.groupby(data_group_name):
        data_i_dedup = data_i.drop_duplicates(NE_col)
        for importance_stat in importance_stats:
            importance_stat_vals = np.log(data_i_dedup.loc[:, importance_stat]+1.)
            importance_stat_bins = np.linspace(importance_stat_vals.min(), importance_stat_vals.max(), N_importance_bins+1)[1:]
            importance_stat_bin_vals = pd.DataFrame(np.digitize(importance_stat_vals, bins=importance_stat_bins), columns=['%s_bin'%(importance_stat)])
            importance_stat_bin_vals.loc[:, NE_col] = data_i_dedup.loc[:, NE_col].values
            data_i = pd.merge(data_i, importance_stat_bin_vals, on=NE_col)
        combined_tag_data_importance.append(data_i)
    combined_tag_data_importance = pd.concat(combined_tag_data_importance, axis=0)

    ## restrict to freq NEs
    ## only include data with frequent entities for better estimation
    if(top_k_NE_min > 0):
        data = []
        for data_name_i, data_i in combined_tag_data_importance.groupby(data_group_name):
            freq_NEs_i = extract_freq_NEs(data_i, dep_var=dep_var, min_count_0=top_k_NE_min, min_count_1=top_k_NE_min)
            data_i_k = data_i[data_i.loc[:, NE_col].isin(freq_NEs_i)]
            data.append(data_i_k)
        data = pd.concat(data, axis=0)
        logging.debug(data.loc[:, 'time_period_type'].value_counts())
    else:
        data = combined_tag_data_importance.copy()
    
    ## fix NE string problems
    data = data.assign(**{NE_col : data.loc[:, NE_col].apply(clean_NE)})
    
    ## fix username to include RT users
    RT_MATCHER = re.compile('(?<=RT @)[A-Za-z0-9_]+(?=:)')
    data = data.assign(**{'username_rt' : data.loc[:, 'txt'].apply(lambda x: RT_MATCHER.search(x).group(0) if RT_MATCHER.search(x) is not None else '')})
    ## fix usernames
    data = data.assign(**{'username' : data.loc[:, 'username'].apply(lambda x: x.split(':')[-1])})
    data = data.assign(**{'username_fixed' : data.apply(lambda x: x.loc['username_rt'] if x.loc['username_rt'] != '' else x.loc['username'], axis=1)})
    
    return data

## user data

def add_user_data(data, user_data_file='../../data/mined_tweets/user_data/user_data.gz', user_var='username_fixed'):
    user_data = pd.read_csv(user_data_file, sep='\t', index_col=False, compression='gzip', usecols=['screen_name', 'followers_count', 'friends_count'])
    user_data.rename(columns={'screen_name':user_var}, inplace=True)
    ## get RT users
    RT_MATCHER = re.compile('(?<=RT @)[A-Za-z0-9_]+(?=:)')
    # handled in load_clean_data
    # print(RT_MATCHER.search('RT @AnderCoop: blah blah'))
#     data = data.assign(**{'username_rt' : data.loc[:, 'txt'].apply(lambda x: RT_MATCHER.search(x).group(0) if RT_MATCHER.search(x) is not None else '')})
    ## how much of the original data is covered?
    data_users = set(data.loc[:, user_var].unique())
#     data_users_rt = set(data.loc[:, 'username_rt'][data.loc[:, 'username_rt'] != ''].unique())
#     data_users = data_users | data_users_rt
    all_users_collected = user_data.loc[:, user_var].unique()
    data_users_with_stats = data_users & set(all_users_collected)
    print('%d/%d users covered'%(len(data_users_with_stats), len(data_users)))
    # restrict to the covered users
    data_user_combined = data[data.loc[:, user_var].isin(all_users_collected)]
    print('%d/%d user data'%(data_user_combined.shape[0], data.shape[0]))
    ## match follower count to original user
#     data_user_combined = data_user_combined.assign(**{'username_original' : data_user_combined.apply(lambda x: x.loc['username_rt'] if x.loc['username_rt'] != '' else x.loc['username'], axis=1)})
    username_original_set = data_user_combined.loc[:, user_var].unique()
    user_data_relevant = user_data[user_data.loc[:, user_var].isin(username_original_set)].drop_duplicates(user_var, inplace=False).loc[:, [user_var, 'followers_count', 'friends_count']]
#     user_data_relevant.rename(columns={user_var:'username_original'}, inplace=True)
    data_user_combined = pd.merge(data_user_combined, user_data_relevant, on=user_var, how='inner')
    ## log-scale
    data_user_combined = data_user_combined.assign(**{
        'followers_log' : np.log(data_user_combined.loc[:, 'followers_count'].values + 1.),
        'friends_log' : np.log(data_user_combined.loc[:, 'friends_count'].values + 1.),
    })
    ## add ratio
    data_user_combined = data_user_combined.assign(**{
        'followers_friends' : data_user_combined.loc[:, 'followers_log'].values - data_user_combined.loc[:, 'friends_log'].values,
    })
    return data_user_combined

## freq data 

def add_freq_data(data, freq_file_name='../../data/mined_tweets/combined_tweet_tag_data_NE_freq.gz'):
    freq = pd.read_csv(freq_file_name, sep='\t', index_col=False, compression='gzip', converters={'TIME':dateutil.parser.parse})
    ## add time data
    metadata_cols = ['TIME', 'DATA_NAME']
    avoid_cols = ['#hash', '<url>', '<_url', 'url_>', '@user', '–_', '-_', '<num>']
    avoid_matcher = re.compile('|'.join(avoid_cols))
    loc_cols = [x for x in freq.columns if x.endswith('_loc') and avoid_matcher.search(x) is None]
    freq_loc = freq.loc[:, loc_cols + metadata_cols]
    display(freq_loc.head())
    ## melt to yield one row per NE/TIME/DATA cooccurrence
    freq_loc = pd.melt(freq_loc, id_vars=metadata_cols, value_vars=loc_cols, value_name='freq', var_name='NE_fixed')
    # replace LOC marker
    freq_loc = freq_loc.assign(**{'NE_fixed' : freq_loc.loc[:, 'NE_fixed'].apply(lambda x: x.replace('_loc', ''))})
    # add previous time
    time_lag = 1
    freq_loc = freq_loc.assign(**{'TIME_JOIN' : freq_loc.loc[:, 'TIME'].apply(lambda x: x + timedelta(days=time_lag))})
#     display(freq_loc.head())
    ## merge with tag data
    time_var = 'date_day'
    data_name_var = 'data_name_fixed'
    data = data.assign(**{
        'TIME_JOIN' : data.loc[:, time_var],
        'DATA_NAME' : data.loc[:, data_name_var],
    })
    freq_anchor_data = pd.merge(data, freq_loc, on=['TIME_JOIN', 'DATA_NAME', 'NE_fixed'], how='left')
    freq_anchor_data.drop(['TIME_JOIN', 'DATA_NAME'], axis=1, inplace=True)
    freq_anchor_data.fillna(0, inplace=True)
    # log-norm frequency data
    freq_anchor_data = freq_anchor_data.assign(**{'freq_log' : freq_anchor_data.loc[:, 'freq'].apply(lambda x: np.log(x+1e-8))})
    # show data
    display(freq_anchor_data.head())
    print('%d/%d samples'%(freq_anchor_data.shape[0], data.shape[0]))
    return freq_anchor_data

def clean_var_name(x):
    return x.replace('-','_').replace('/', '_').replace("'", '_')

def fit_evaluate_model(data, ind_vars, dep_var, test=0.1, k=10, balance=False):
    """
    Fit and evaluate LR model based on ability
    to predict dep_var. 
    We are interested in (1) predictive power and (2) deviance from null model.
    
    :param data: prediction data
    :param ind_vars: independent vars
    :param dep_var: dependent var
    :param test: test percent
    :param k: k_fold classification count
    :param balance: balance data on minority class
    :returns model_results:: model results from fit on full data
    :returns predict_acc:: accuracy on test data from k-fold cross-validation
    """
    np.random.seed(123)
    formula = '%s ~ %s'%(dep_var, ' + '.join(ind_vars))
    logging.debug('formula: %s'%(formula))
#     logging.debug(data.loc[:, 'NE_fixed'].head())
    ## regular fit/statistics
    model = glm(formula=formula, data=data, family=Binomial())
    model_results = model.fit()
    logging.debug(model_results.summary())
    if(balance):
        data.loc[:, dep_var] = data.loc[:, dep_var].astype(int)
        dep_var_counts = data.loc[:, dep_var].value_counts()
        N_min_class = dep_var_counts.iloc[-1]
        data_balanced = pd.concat([data_c.loc[np.random.choice(data_c.index, N_min_class, replace=False), :] for c, data_c in data.groupby(dep_var)], axis=0)
        data = data_balanced.copy()
#     logging.debug(data.loc[:, 'NE_fixed'].head())
            
    ## k-fold cross validation
    # convert categorical vars to usable format
    reg_data = data.copy()
    cat_var_matcher = re.compile('C\((.+)\)')
    ind_vars_cat = [cat_var_matcher.search(x).group(1) for x in ind_vars if cat_var_matcher.search(x) is not None]
    if(len(ind_vars_cat) > 0):
        ind_var_cat_vals = []
    #     logging.debug(reg_data.loc[:, ind_vars_cat].head())
        for ind_var_cat in ind_vars_cat:
            ind_var_unique_vals = list(reg_data.loc[:, ind_var_cat].unique())
    #             logging.debug(unique_val)
            reg_data = reg_data.assign(**{clean_var_name(x):(reg_data.loc[:, ind_var_cat]==x).astype(int) for x in ind_var_unique_vals})
            # fix bad strings
            ind_var_unique_vals = [clean_var_name(x) for x in ind_var_unique_vals]
            ind_var_cat_vals += ind_var_unique_vals
            reg_data.drop(ind_var_cat, axis=1, inplace=True)
    #     logging.debug('data cols %s'%(str(reg_data.columns)))
        ind_vars_full = (set(ind_vars) - set(['C(%s)'%(x) for x in ind_vars_cat])) | set(ind_var_cat_vals)
        formula_full = '%s ~ %s'%(dep_var, ' + '.join(ind_vars_full))
    else:
        formula_full = '%s ~ %s'%(dep_var, ' + '.join(ind_vars))
#     logging.debug('formula full => %s'%(formula_full))
    kfold = KFold(n_splits=k, shuffle=True)
    predict_acc = []
    reg_data.loc[:, dep_var] = reg_data.loc[:, dep_var].astype(int)
    for train_idx, test_idx in kfold.split(reg_data):
        data_train = reg_data.iloc[train_idx, :]
        data_test = reg_data.iloc[test_idx, :]
#         logging.debug('train data %s'%(str(data_train.columns)))
        model_i = logit(formula=formula_full, data=data_train)
#         model_i = logit(endog=train_data.loc[:, dep_var], exog=train_data.loc[:, ind_vars])
        model_i_results = model_i.fit(full_output=False, disp=True)
        model_i_results.predict(data_test)
        pred_vals_i = np.array([int(x > 0.5) for x in model_i_results.predict(data_test)])
        y = data_test.loc[:, dep_var].astype(int)
#         predict_results_i = 1 - ((y - pred_vals_i) / len(y))
        predict_results_i = (y == pred_vals_i)
        predict_acc_i = np.mean(predict_results_i)
        predict_acc.append(predict_acc_i)
    return model_results, predict_acc

def pandas_df_to_markdown_table(df, out_file):
    from IPython.display import Markdown, display
    fmt = ['---' for i in range(len(df.columns))]
    df_fmt = pd.DataFrame([fmt], columns=df.columns)
    df_formatted = pd.concat([df_fmt, df])
    out_file.write(df_formatted.to_csv(sep="|", index=False))

def predict_results_coeffs(data, data_name, out_dir, dep_var='has_descriptor'):
    """
    Run prediction to compute accuracy/deviance and coefficients/significance.
    """
    data_results = []
    data_coeffs = []
    coeff_names_to_track = set(['since_form', 'since_landfall', 'since_dissipation', 'C(time_period_type)'])
    ind_var_sets = [
        ['C(NE_fixed)'], 
        ['C(NE_fixed)', 'since_form'],
        ['C(NE_fixed)', 'since_form', 'since_landfall', 'since_dissipation'],
        ['C(NE_fixed)', 'C(time_period_type)'],
        ['since_form'],
        ['since_form', 'since_landfall', 'since_dissipation'],
        ['C(time_period_type)'],
        ['max_pop_bin', 'max_names_bin'],
        ['max_pop_bin', 'max_names_bin', 'since_form'],
        ['max_pop_bin', 'max_names_bin', 'since_form', 'since_landfall', 'since_dissipation'],
        ['max_pop_bin', 'max_names_bin', 'C(time_period_type)'],
    ]
    cat_coeff_names_to_track = ['time_period_type']
    if(len(cat_coeff_names_to_track) > 0):
        cat_coeff_matcher = re.compile('|'.join(cat_coeff_names_to_track))
    else:
        cat_coeff_matcher = re.compile('.*')
    test_pct = 0.9
#     data_name = 'joint'
    for ind_var_set in ind_var_sets:
        logging.debug('testing ind_vars = %s'%(','.join(ind_var_set)))
        model_results, predict_acc = fit_evaluate_model(data, ind_var_set, dep_var, test=test_pct, balance=True)
        logging.debug('data=%s, mean_acc=%.3f, dev=%d'%(data_name, np.mean(predict_acc), model_results.deviance))
        data_results.append(['+'.join(ind_var_set), np.mean(predict_acc), model_results.deviance])
        if(len(coeff_names_to_track & set(ind_var_set)) > 0):
            data_coeffs.append([['+'.join(ind_var_set), i, x, model_results.pvalues.loc[i]] for i, x in zip(model_results.params.index, model_results.params) if i in ind_var_set or cat_coeff_matcher.search(i) is not None])
    data_results = pd.DataFrame(data_results, columns=['ind_vars', 'acc_mean', 'deviance'])
    data_coeffs = pd.DataFrame(np.vstack(data_coeffs), columns=['ind_vars', 'var_name', 'coeff', 'p_val'])
    
    ## output to file
    result_out_file_name = os.path.join(out_dir, '%s_%s_data_results.md'%(data_name, dep_var))
    float_cols = ['acc_mean', 'deviance']
    for x in float_cols:
        data_results.loc[:, x] = data_results.loc[:, x].apply(lambda x: '%.3f'%(x))
    with open(result_out_file_name, 'w') as out_file:
        pandas_df_to_markdown_table(data_results, out_file)
    ## get coefficients, p-vals for time and importance vars
    coeff_out_file_name = os.path.join(out_dir, '%s_%s_data_results.md'%(data_name, dep_var))
    coeff_float_cols = ['coeff', 'p_val']
    logging.debug('processing results data %s'%(data_name))
    # format cols
    for x in coeff_float_cols:
        data_coeffs.loc[:, x] = data_coeffs.loc[:, x].apply(lambda x: '%.3f'%(float(x)))
#     data_coeffs.drop(['data_name'], axis=1, inplace=True)
    with open(coeff_out_file_name, 'w') as out_file:
        pandas_df_to_markdown_table(data_coeffs, out_file)
    
def main():
    parser = ArgumentParser()
    parser.add_argument('--combined_data', default='../../data/mined_tweets/combined_tweet_tag_data_NE_flat.gz')
#     parser.add_argument('--combined_data', default='../../data/mined_tweets/combined_location_phrase_tweet_tag_data_NE_flat.gz')
    parser.add_argument('--out_dir', default='../../output/predict_results/')
    parser.add_argument('--dep_var', default='has_descriptor')
#     parser.add_argument('--dep_var', default='max_population_anchor')
    parser.add_argument('--ind_vars', default=[])
    args = vars(parser.parse_args())
    
    pred_data_name = os.path.basename(args['combined_data']).replace('.gz', '')
    log_file_name = '../../output/regression_results/predict_descriptor_in_twitter_%s_%s.txt'%(args['dep_var'], pred_data_name)
    if(os.path.exists(log_file_name)):
        os.remove(log_file_name)
    logging.basicConfig(filename=log_file_name, level=logging.DEBUG)
    
    out_dir = os.path.join(args['out_dir'], pred_data_name)
    if(not os.path.exists(out_dir)):
        os.mkdir(out_dir)
    
    ## load data
    ## clean data etc.
    combined_data = load_clean_data(args['combined_data'], dep_var=args['dep_var'])
    logging.debug('%d data samples'%(combined_data.shape[0]))
    
    ## TODO: followers, frequency
    
    ## predict!
    ## joint data
    predict_results_coeffs(combined_data, 'joint', out_dir, dep_var=args['dep_var'])
    
    ## separate data
    data_group_name = 'data_name_fixed'
    for data_name_i, data_i in combined_data.groupby(data_group_name):
        predict_results_coeffs(combined_data, data_name_i, out_dir, dep_var=args['dep_var'])
    
    ## TODO: add audience information, re-run predictions
    
if __name__ == '__main__':
    main()