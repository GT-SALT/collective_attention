"""
Test a range of L2 weights for 
fixed-effect regularized regression.
The goal is to minimize log-likelihood in
a held-out test set.
"""
from argparse import ArgumentParser
import logging
import os
import numpy as np
import pandas as pd
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families.family import Binomial
from statsmodels.genmod.families.links import logit
from math import ceil, floor
from sklearn.preprocessing import StandardScaler
from ast import literal_eval
from datetime import datetime, timedelta
import dateutil
import sys
if('..' not in sys.path):
    sys.path.append('..')
from data_processing.data_helpers import fix_timezone, round_to_day

np.random.seed(123)

def logit_cdf(X):
    return 1 / (1 + np.exp(-X))

## log likelihood
def compute_log_likelihood(params, Y, X):
    q = 2 * Y - 1
    ll = np.sum(np.log(logit_cdf(q * np.dot(X, params))))
    return ll

def load_facebook_data():
    group_data = pd.read_csv('../../data/facebook-maria/combined_group_data_es_tagged_valid_anchor_group_contain.tsv', sep='\t', index_col=False, converters={'subtree' : literal_eval, 'tree' : literal_eval})
    # importance: NE frequency
    NE_var = 'NE_fixed'
    NE_counts = group_data.groupby(NE_var).apply(lambda x: x.shape[0]).reset_index().rename(columns={0 : 'NE_count'})
    group_data = pd.merge(group_data, NE_counts, on=NE_var)
    # information: post length
    group_data = get_text_with_no_context(group_data)
    txt_var = 'status_message'
    no_context_txt_var = '%s_no_context'%(txt_var)
    group_data = group_data.assign(**{
        'txt_len_norm' : group_data.loc[:, no_context_txt_var].apply(lambda x: np.log(len(x)+1))
    })
    # commitment: number of posts per author per group
    author_var = 'status_author_id'
    group_var = 'group_name'
    author_group_counts = group_data.groupby([author_var, group_var]).apply(lambda x: x.shape[0]).reset_index().rename(columns={0 : 'author_group_count'})
    group_data = pd.merge(group_data, author_group_counts, on=[author_var, group_var])
    # audience: group size
    group_counts = group_data.groupby(group_var).apply(lambda x: x.loc[:, author_var].nunique()).reset_index().rename(columns={0 : 'group_size'})
    group_data = pd.merge(group_data, group_counts, on=group_var)
    ## Z-norm scalar vars
    scaler = StandardScaler()
    group_data_reg = group_data.copy()
    # add intercept
    group_data_reg = group_data_reg.assign(**{
        'intercept' : 1.
    })
    scalar_vars = ['NE_count', 'txt_len_norm', 'author_group_count', 'group_size']
    for v in scalar_vars:
        group_data_reg = group_data_reg.assign(**{
            v : scaler.fit_transform(group_data_reg.loc[:, v].values.reshape(-1,1))
        })
    group_data_reg = group_data_reg.assign(**{
        'group_contains_NE' : group_data_reg.loc[:, 'group_contains_NE'].astype(int)
    })
    min_author_count = 20
    min_NE_count = 20
    min_group_count = 20
    author_var = 'status_author_id'
    NE_var = 'NE_fixed'
    group_var = 'group_name'
    min_counts = [min_author_count, min_NE_count, min_group_count]
    cat_vars = ['status_author_id', 'NE_fixed', 'group_name']
    for cat_var, min_count in zip(cat_vars, min_counts):
        cat_counts = group_data_reg.loc[:, cat_var].value_counts()
        group_data_reg = group_data_reg.assign(**{
            '%s_cap'%(cat_var) : group_data_reg.loc[:, cat_var].apply(lambda x: 'RARE' if cat_counts.loc[x] < min_count else x)
        })
    return group_data_reg
    
def load_twitter_data():
    anchor_data = pd.read_csv('../../data/mined_tweets/combined_tweet_tag_data_NE_flat_parsed_anchor_NE_peak_times_consistent_authors.gz', sep='\t', index_col=False, compression='gzip')
    return anchor_data
    # old cleaning code 
#     anchor_data = pd.read_csv('../../data/mined_tweets/combined_tweet_tag_data_NE_flat_parsed_anchor.gz', sep='\t', index_col=False, compression='gzip', converters={'date' : dateutil.parser.parse})
#     anchor_data = anchor_data.assign(**{'username' : anchor_data.loc[:, 'username'].apply(lambda x: x.split(':')[-1])})
#     ## add time var
#     date_var = 'date'
#     full_data = full_data.assign(**{time_var : full_data.loc[:, time_var].apply(lambda x: fix_timezone(x))})
#     anchor_data = anchor_data.assign(**{time_var : anchor_data.loc[:, time_var].apply(lambda x: fix_timezone(x))})
#     time_var = 'date'
#     round_time_var = '%s_day'%(time_var)
#     full_data = full_data.assign(**{round_time_var : full_data.loc[:, time_var].apply(lambda x: round_to_day(x))})
#     anchor_data = anchor_data.assign(**{round_time_var : anchor_data.loc[:, time_var].apply(lambda x: round_to_day(x))})
#     data_name_var = 'data_name_fixed'
#     NE_var = 'NE_fixed'
#     min_NE_count = 20
#     anchor_data_NE_peak = []
#     for name_i, data_i in anchor_data.groupby(data_name_var):
#     #     print('data=%s'%(name_i))
#         NE_counts = data_i.loc[:, NE_var].value_counts()
#     #     display(NE_counts)
#         NE_consistent = NE_counts[NE_counts >= min_NE_count].index.tolist()
#         data_i_NE = data_i[data_i.loc[:, NE_var].isin(NE_consistent)]
#         data_i_NE_peak_times = data_i_NE.groupby(NE_var).apply(lambda x: compute_peak_time(x, time_var=round_time_var, verbose=False))
#         display(data_i_NE_peak_times)
#         data_i_NE_peak_times = pd.DataFrame(data_i_NE_peak_times).rename(columns={0:'NE_peak_date'})
#         data_i_NE_peak_times = data_i_NE_peak_times.assign(**{'NE_fixed' : data_i_NE_peak_times.index})
#         data_i_NE_peak_times.index = np.arange(data_i_NE_peak_times.shape[0])
#         display(data_i_NE_peak_times.head())
#         data_i = pd.merge(data_i, data_i_NE_peak_times, on='NE_fixed', how='inner')
#         anchor_data_NE_peak.append(data_i)
#     anchor_data_NE_peak = pd.concat(anchor_data_NE_peak, axis=0)
#     ## per-data peak times
#     # visualize aggregate probability (bootstraps?) and per-NE probability
#     data_name_var = 'data_name_fixed'
#     date_var = 'date_day'
#     peak_var = 'data_peak_date'
#     anchor_var = 'anchor'
#     date_buffer = timedelta(days=1)
#     ## assign pre/during/post
#     anchor_data = anchor_data.assign(**{'pre_peak' : (anchor_data.loc[:, date_var] <= anchor_data.loc[:, peak_var] - date_buffer).astype(int)})
#     anchor_data = anchor_data.assign(**{'during_peak' : ((anchor_data.loc[:, date_var] >= anchor_data.loc[:, peak_var] - date_buffer) & (anchor_data.loc[:, date_var] <= anchor_data.loc[:, peak_var] + date_buffer)).astype(int)})
#     anchor_data = anchor_data.assign(**{'post_peak' : (anchor_data.loc[:, date_var] >= anchor_data.loc[:, peak_var] + date_buffer).astype(int)})
#     ## per-NE peak times
#     ## assign pre/post
#     data_name_var = 'data_name_fixed'
#     date_var = 'date_day'
#     peak_var = 'NE_peak_date'
#     anchor_var = 'anchor'
#     date_buffer = timedelta(days=1)
#     anchor_data_NE_peak = anchor_data_NE_peak.assign(**{'pre_peak' : (anchor_data_NE_peak.loc[:, date_var] <= anchor_data_NE_peak.loc[:, peak_var] - date_buffer).astype(int)})
#     anchor_data_NE_peak = anchor_data_NE_peak.assign(**{'post_peak' : (anchor_data_NE_peak.loc[:, date_var] >= anchor_data_NE_peak.loc[:, peak_var] + date_buffer).astype(int)})
#     min_author_count = 10
#     author_var = 'username'
#     data_name_var = 'data_name_fixed'
#     anchor_data_NE_peak_filter = []
#     for name_i, data_i in anchor_data_NE_peak.groupby(data_name_var):
#         author_counts = data_i.loc[:, author_var].value_counts()
#         NE_counts = data_i.loc[:, NE_var].value_counts()
#         # filter authors
#         data_i = data_i.assign(**{author_var : data_i.loc[:, author_var].apply(lambda x: 'RARE_AUTHOR' if author_counts.loc[x] < min_author_count else x)})
#         # filter NEs - we already filtered when preprocessing
#     #     data_i = data_i.assign(**{NE_var : data_i.loc[:, NE_var].apply(lambda x: 'RARE_NE' if NE_counts.loc[x] < min_NE_count else x)})
#         anchor_data_NE_peak_filter.append(data_i)
#     anchor_data_NE_peak_filter = pd.concat(anchor_data_NE_peak_filter, axis=0)
#     return anchor_data_NE_peak_filter

def test_weights(data, dep_var, cat_vars, scalar_vars, l2_weights):
    indep_formula = ' + '.join(['C(%s)'%(cap_cat_var) for cap_cat_var in cap_cat_vars] + scalar_vars)
    formula = '%s ~ %s'%(dep_var, indep_formula)
    # convert raw data to exogenous data
    # need to do this to force train/test
    # to have same features
    data_rand = data.copy()
    np.random.shuffle(data_rand.values)
    model_dummy = GLM.from_formula(formula, data_rand, family=Binomial(link=logit()))
    exog = model_dummy.exog
    exog_names = model_dummy.exog_names
    endog = model_dummy.endog
    # generate cross validation folds
    cross_val_folds = 10
    N = data_rand.shape[0]
    cross_val_chunk_size = float(N) / cross_val_folds
    cross_val_fold_train_idx = [list(range(int(floor(i*cross_val_chunk_size)), int(ceil((i+1)*cross_val_chunk_size)))) for i in range(cross_val_folds)]
    cross_val_fold_test_idx = [list(range(0, int(ceil(i*cross_val_chunk_size)))) + list(range(int(floor((i+1)*cross_val_chunk_size)), N)) for i in range(cross_val_folds)]
    weight_likelihoods = []
    for l2_weight in l2_weights:
        print('testing weight = %.3f'%(l2_weight))
        likelihoods_l2 = []
        for i, (train_idx_i, test_idx_i) in enumerate(zip(cross_val_fold_train_idx, cross_val_fold_test_idx)):
            print('fold %d'%(i))
            train_XY = data_rand.iloc[train_idx_i, :]
            test_X = exog[test_idx_i, :]
            test_Y = endog[test_idx_i]
            # fit model
            model_i = GLM.from_formula(formula, train_XY, family=Binomial(link=logit()))
            model_res_i = model_i.fit_regularized(maxiter=max_iter, method='elastic_net', alpha=l2_weight, L1_wt=0.)
            # add 0 params for missing coefficients
            # to match X shape
            model_res_i.params = model_res_i.params.loc[exog_names].fillna(0, inplace=False)
            # score test data
            likelihood_i = compute_log_likelihood(model_res_i.params, test_Y, test_X)
            likelihoods_l2.append(likelihood_i)
        weight_likelihoods.append(likelihoods_l2)
    weight_likelihoods = pd.DataFrame(np.array(weight_likelihoods), index=l2_weights)
    mean_weight_likelihoods = weight_likelihoods.mean(axis=0)
    return mean_weight_likelihoods

def get_text_with_no_context(data, id_var='status_id', txt_var='status_message', subtree_var='subtree', tree_var='tree', context_var='anchor'):
    """
    Replace context from text and return clean text.
    """
    no_context_txt_var = '%s_no_context'%(txt_var)
    data_no_context_txt = []
    for id_i, data_i in data.groupby(id_var):
        txt_i = data_i.loc[:, txt_var].iloc[0]
        # replace context in txt_i
        txt_i_clean = txt_i
        for idx_j, NE_data_j in data_i.iterrows():
            if(NE_data_j.loc[context_var]==1):
    #             print('clean txt before: %s'%(txt_i_clean))
                tree_txt_j = ' '.join([token[0] for token in NE_data_j.loc[tree_var]])
                subtree_txt_j = ' '.join([token[0] for token in NE_data_j.loc[subtree_var]])
                txt_i_clean = txt_i_clean.replace(tree_txt_j, '')
                txt_i_clean = txt_i_clean.replace(subtree_txt_j, '')
        data_i = data_i.assign(**{
            no_context_txt_var : data_i.apply(lambda x: txt_i_clean if x.loc[context_var]==1 else txt_i, axis=1)
        })
        data_no_context_txt.append(data_i)
    data_no_context_txt = pd.concat(data_no_context_txt, axis=0)
    return data_no_context_txt
    
def main():
    parser = ArgumentParser()
    parser.add_argument('--l2_weights', default=[0., 0.001, 0.01, 0.1, 1.])
    parser.add_argument('--data_type', default='facebook')
    parser.add_argument('--out_dir', default='../../output')
#     parser.add_argument('--data_type', default='twitter')
    args = vars(parser.parse_args())
    logging_file = '../../output/test_L2_weights_fixed_effect_regression_%s.txt'%(args['data_type'])
    if(os.path.exists(logging_file)):
        os.remove(logging_file)
    logging.basicConfig(filename=logging_file, level=logging.DEBUG)
    
    ## load data
    if(args['data_type']=='facebook'):
        ## TODO: add clean_facebook_data function in same way as Twitter
        ## then just load clean data??
        data = load_facebook_data()
        ## define vars
        dep_var = 'anchor'
        raw_cat_vars = ['NE_fixed', 'status_author_id', 'group_name']
        cat_vars = ['%s_cap'%(raw_cat_var) for raw_cat_var in raw_cat_vars]
        scalar_vars = ['NE_count', 'txt_len_norm', 'author_group_count', 'group_size', 'group_contains_NE']
    elif(args['data_type'] == 'twitter'):
        data = load_twitter_data()
        ## define vars
        dep_var = 'anchor'
        cat_vars = ['username', 'NE_fixed', 'data_name_fixed']
        scalar_vars = ['during_peak', 'post_peak']
    
    ## test weights
    weight_likelihoods = test_weights(data, dep_var, cat_vars, scalar_vars, args['l2_weights'])
    
    ## save to file
    out_file = os.path.join(args['out_dir'], 'L2_weight_test_%s.tsv'%(args['data_type']))
    weight_likelihoods.to_csv(out_file, sep='\t', index=False)
    
if __name__ == '__main__':
    main()