"""
Run regression on permuted peak date times.
"""
import numpy as np
import pandas as pd
## need data
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families.family import Binomial
from statsmodels.genmod.families.links import logit
from model_helpers import compute_err_data
from argparse import ArgumentParser
import logging
import os
from sklearn.preprocessing import StandardScaler
import dateutil
from datetime import timedelta

def update_peaks_fit_regression(data, NE_date_ranges, NE_var, data_name_var, round_date_var, peak_date_var, peak_date_buffer, scalar_vars, formula, max_iter, l2_weight, regression_type):
    """
    Randomly update peak times and fit regression.
    """
    NE_peak_dates_i = NE_date_ranges.apply(lambda x: np.random.choice(x, 1)[0])
    NE_peak_dates_i_df = NE_peak_dates_i.reset_index().rename(columns={0 : peak_date_var})
#         data_peak_dates_i = data_date_ranges.apply(lambda x: np.random.choice(x, 1)[0]).reset_index().rename(columns={0 : peak_date_var})
    data_i = pd.merge(data, NE_peak_dates_i_df, on=[NE_var, data_name_var], how='inner')
    # reassign peaks
    data_i = data_i.assign(**{
        'pre_peak' : (data_i.loc[:, round_date_var] <= data_i.loc[:, peak_date_var] - peak_date_buffer).astype(int),
        'post_peak' : (data_i.loc[:, round_date_var] >= data_i.loc[:, peak_date_var] + peak_date_buffer).astype(int),
        'during_peak' : ((data_i.loc[:, round_date_var] > data_i.loc[:, peak_date_var] - peak_date_buffer) & 
                         (data_i.loc[:, round_date_var] < data_i.loc[:, peak_date_var] + peak_date_buffer)).astype(int),
    })
    # add days since post-peak
    data_i = data_i.assign(**{
        'since_peak' : data_i.loc[:, 'post_peak'] * (data_i.loc[:, round_date_var] - data_i.loc[:, peak_date_var])
    })
    # Z-norm all scalar vars
    scaler = StandardScaler()
    for v in scalar_vars:
        data_i = data_i.assign(**{v : scaler.fit_transform(data_i.loc[:, v].values.reshape(-1,1))})
    model_full = GLM.from_formula(formula, data_i, family=Binomial(link=logit()))
    logging.debug('%d/%d/%d pre/during/post data'%(data_i.loc[:, 'pre_peak'].sum(), data_i.loc[:, 'during_peak'].sum(), data_i.loc[:, 'post_peak'].sum()))
    if(regression_type == 'regularized_logit'):
        model_res_full = model_full.fit_regularized(maxiter=max_iter, method='elastic_net', alpha=l2_weight, L1_wt=0.0)
        model_res_full_err = compute_err_data(model_res_full)
        err = model_res_full_err.loc[:, 'SE']
    else:
        model_res_full = model_full.fit()
        err = model_res_full.bse
    params = model_res_full.params
    return params, err, NE_peak_dates_i

def permute_fit_regression(data, NE_date_ranges, peak_date_buffer, dep_var='anchor', cat_vars=[], binary_vars=[], scalar_vars=[], permute_iters=20, max_iter=20, l2_weight=0.01, regression_type='regularized_logit', parallel=False):
    """
    Permute NE peak dates and fit regression.
    """
    peak_date_var = 'peak_date'
    round_date_var = 'date_day'
    NE_var = 'NE_fixed'
    data_name_var = 'data_name_fixed'
    cat_var_str = '+'.join('C(%s)'%(x) for x in cat_vars)
    binary_var_str = '+'.join(binary_vars)
    scalar_var_str = '+'.join(scalar_vars)
    formula = '%s ~ %s + %s + %s'%(dep_var, cat_var_str, scalar_var_str, binary_var_str)
    permute_params = []
    permute_err = []
    permute_peak_dates = []
    ## TODO: make parallel
    if(parallel):
        pass
    else:
        for i in range(permute_iters):
            permute_params_i, permute_err_i, permute_dates_i = update_peaks_fit_regression(data, NE_date_ranges, NE_var, data_name_var, round_date_var, peak_date_var, peak_date_buffer, scalar_vars, formula, max_iter, l2_weight, regression_type)
            permute_params.append(permute_params_i)
            permute_err.append(permute_err_i)
            permute_peak_dates.append(permute_dates_i)
            logging.debug('permute iter=%d'%(i))
    ## compute mean, standard error
    permute_params = pd.concat(permute_params, axis=1)
    permute_param_mean = permute_params.mean(axis=1)
    # compute err from model params (is this right?)
    permute_param_sd = permute_params.std(axis=1)
    permute_param_sd_err = permute_param_sd
    # compute err from model err
    permute_param_err = pd.concat(permute_err, axis=1).mean(axis=1)
    permute_param_df = pd.concat([permute_param_mean, permute_param_sd_err], axis=1)
    permute_param_df.columns = ['estimate', 'err']
    # save dates
    permute_peak_dates = pd.concat(permute_peak_dates, axis=1)
    return permute_param_df, permute_peak_dates

np.random.seed(123)
def main():
    parser = ArgumentParser()
    # original data
    parser.add_argument('--clean_data', default='../../data/mined_tweets/combined_tweet_tag_data_NE_flat_parsed_anchor_NE_peak_times_consistent_authors.gz')
    # power user data
#     parser.add_argument('--clean_data', default='../../data/mined_tweets/combined_tweet_tag_data_NE_flat_parsed_anchor_prior_author_stats.gz')
    parser.add_argument('--cat_vars', type=list, default=['data_name_fixed', 'username', 'NE_fixed'])
    # original data: time vars
#     parser.add_argument('--binary_vars', type=list, default=['during_peak', 'post_peak', 'has_URL'])
    parser.add_argument('--binary_vars', type=list, default=['during_peak', 'post_peak', 'has_URL'])
    # original data: no time vars
#     parser.add_argument('--binary_vars', type=list, default=['has_URL'])
    # original data: text length 
    # TODO: make prior text length? at least for power user regression, otherwise could be conditioning on dependent variable
#     parser.add_argument('--scalar_vars', type=list, default=['txt_len_norm']) 
    # original data: text length and since peak
    parser.add_argument('--scalar_vars', type=list, default=['txt_len_norm', 'since_start'])
#     parser.add_argument('--scalar_vars', type=list, default=['since_start'])
    # original data: prior NE count
#     parser.add_argument('--scalar_vars', type=list, default=['txt_len_norm', 'NE_count_prior'])
    # power user data
#     parser.add_argument('--scalar_vars', type=list, default=['post_count', 'NE_count', 'prior_retweets', 'prior_favorites', 'pre_peak', 'post_peak'])
    # original data: prior NE count
#     parser.add_argument('--log_vars', default=['NE_count_prior'])
    # power user
#     parser.add_argument('--log_vars', default=['post_count', 'NE_count', 'prior_retweets', 'prior_favorites'])
    # regression type
    parser.add_argument('--regression_type', default='regularized_logit')
#     parser.add_argument('--regression_type', default='logit')
    parser.add_argument('--dep_var', default='anchor')
    # process in parallel => faster??
#     parser.add_argument('--parallel', type=bool, default=True)
    parser.add_argument('--parallel', type=bool, default=False)
    parser.add_argument('--permute_iters', type=int, default=50)
    args = vars(parser.parse_args())
    
    ## set up vars
    cat_vars = args.get('cat_vars')
    binary_vars = args.get('binary_vars')
    scalar_vars = args.get('scalar_vars')
    log_vars = args.get('log_vars')
    if(cat_vars is None):
        cat_vars = []
    if(binary_vars is None):
        binary_vars = []
    if(scalar_vars is None):
        scalar_vars = []
    if(log_vars is None):
        log_vars = []

    cat_var_str = '+'.join('C(%s)'%(x) for x in cat_vars)
    binary_var_str = '+'.join(binary_vars)
    scalar_var_str = '+'.join(scalar_vars)
    formula = '%s ~ %s + %s + %s'%(args['dep_var'], cat_var_str, scalar_var_str, binary_var_str)
    
    ## set up logging
    logging_file = '../../output/run_anchor_regression_permute_%s.txt'%(formula)
    if(os.path.exists(logging_file)):
        os.remove(logging_file)
    logging.basicConfig(filename=logging_file, level=logging.DEBUG)
        
    ## load data
    round_date_var = 'date_day'
    peak_date_var = 'peak_date'
    anchor_data = pd.read_csv(args['clean_data'], sep='\t', index_col=False, compression='gzip', converters={round_date_var : dateutil.parser.parse, peak_date_var : dateutil.parser.parse})
    ## log-scale vars
    v_smooth = 1
    for v in log_vars:
        anchor_data = anchor_data.assign(**{v : np.log(anchor_data.loc[:, v] + v_smooth)})
    
    ## compute NE ranges
    NE_var = 'NE_fixed'
    data_name_var = 'data_name_fixed'
    date_range_buffer = 1
    NE_date_ranges = anchor_data.groupby([data_name_var, NE_var]).apply(lambda x: [x.loc[:, round_date_var].min() + timedelta(days=date_range_buffer*i) for i in range(int((x.loc[:, round_date_var].max() - x.loc[:, round_date_var].min()).days / date_range_buffer))])
    
    ## run regression
    ## fit model with full data 
    ## regularization with best hyperparam in terms of LL
    ## scripts/data_processing/compare_anchor_probability_before_after_peak_in_twitter.ipynb#Fixed-effect-regression,-regularized
    l2_weight = 0.01
    max_iter = 20
#     permute_iters = 20
    permute_params = []
    permute_peak_dates = []
    peak_date_buffer = timedelta(days=1)
    anchor_data_no_peaks = anchor_data.drop(peak_date_var, axis=1)
    permute_params, permute_dates = permute_fit_regression(anchor_data_no_peaks, NE_date_ranges, peak_date_buffer=peak_date_buffer, dep_var=args['dep_var'], cat_vars=cat_vars, binary_vars=binary_vars, scalar_vars=scalar_vars, permute_iters=args['permute_iters'], max_iter=max_iter, l2_weight=l2_weight, regression_type=args['regression_type'])
    
    ## write to file
    out_dir = '../../output/'
    res_out_file = os.path.join(out_dir, 'permute_anchor_%s_iters%d_output_%s.tsv'%(args['regression_type'], args['permute_iters'], formula.replace(' ','')))
    permute_params.to_csv(res_out_file, sep='\t', index=True)
    peak_date_out_file = os.path.join(out_dir, 'permute_anchor_regression_peak_dates.tsv')
    permute_dates.to_csv(peak_date_out_file, sep='\t', index=True)
    
if __name__ == '__main__':
    main()